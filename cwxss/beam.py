"""CTC prefix beam search, with what we know about CW as a prior.

Greedy decoding takes the most likely class at each frame independently and
throws away everything else the model said. That is cheap and it is wasteful:
the model may rate K6XSS and K6XSZ almost equally at the final character, and
greedy picks whichever won by a hair with no reference to the fact that one of
them is a word and the other is not.

Prefix beam search keeps several candidate readings alive and scores each by
the total probability of every frame alignment that produces it. Adding a
language model on top -- here, the vocabulary operators actually send -- lets
evidence about the language settle what the acoustics cannot.

The standard reference is Graves' CTC decoding and Hannun's prefix beam search
write-up; this is that algorithm with a small domain lexicon in place of a
general language model, which is the right size of prior for a vocabulary of a
few hundred words.
"""
import math
from collections import defaultdict

from lexicon import WORDS, frequency_rank

NEG_INF = -float("inf")
VOCAB = set(WORDS)
# Prefixes of real words, so a partial word is not punished for being partial.
PREFIXES = set()
for _w in WORDS:
    for _i in range(1, len(_w) + 1):
        PREFIXES.add(_w[:_i])


def _logsumexp(a, b):
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def word_bonus(prefix, alphabet, lm_weight):
    """How much the language likes the last token of this reading.

    Rewarded for being a real word, mildly rewarded for still being able to
    become one, and penalised for being neither. Without the middle case the
    search abandons every long word halfway through.
    """
    if lm_weight <= 0 or not prefix:
        return 0.0
    tail = prefix.split(" ")[-1]
    if not tail:
        return 0.0
    if tail in VOCAB:
        # common words are likelier than rare ones, but only slightly
        rank = frequency_rank(tail)
        return lm_weight * (1.0 + 0.3 * math.exp(-rank / 60.0))
    if tail in PREFIXES:
        return lm_weight * 0.25
    if len(tail) > 2:
        return -lm_weight * 0.5
    return 0.0


def decode(logprobs, alphabet, blank=0, beam_width=16, lm_weight=0.6,
           prune=-9.0):
    """CTC log-probabilities to text.

    `logprobs` is (frames, classes). `alphabet` maps class index to character,
    with index `blank` reserved.
    """
    beams = {"": (0.0, NEG_INF)}          # prefix -> (log p_blank, log p_non_blank)

    for frame in logprobs:
        # Only consider classes with meaningful probability. A CW alphabet is
        # small, but most of it is near zero at any instant and scoring all of
        # it every frame is most of the cost.
        cands = [k for k in range(len(frame)) if frame[k] > prune]
        if blank not in cands:
            cands.append(blank)
        nxt = defaultdict(lambda: (NEG_INF, NEG_INF))

        for prefix, (pb, pnb) in beams.items():
            total = _logsumexp(pb, pnb)
            for k in cands:
                p = float(frame[k])
                if k == blank:
                    b, nb = nxt[prefix]
                    nxt[prefix] = (_logsumexp(b, total + p), nb)
                    continue
                ch = alphabet.get(k, "")
                if not ch:
                    continue
                last = prefix[-1] if prefix else ""
                if ch == last:
                    # repeating a character extends it unless a blank came
                    # between, which is what separates "BOB" from "BOOB"
                    b, nb = nxt[prefix]
                    nxt[prefix] = (b, _logsumexp(nb, pnb + p))
                    ext = prefix + ch
                    b2, nb2 = nxt[ext]
                    nxt[ext] = (b2, _logsumexp(nb2, pb + p + word_bonus(
                        ext, alphabet, lm_weight)))
                else:
                    ext = prefix + ch
                    b2, nb2 = nxt[ext]
                    nxt[ext] = (b2, _logsumexp(nb2, total + p + word_bonus(
                        ext, alphabet, lm_weight)))

        beams = dict(sorted(nxt.items(),
                            key=lambda kv: -_logsumexp(*kv[1]))[:beam_width])

    best = max(beams.items(), key=lambda kv: _logsumexp(*kv[1]))[0]
    return best.strip()
