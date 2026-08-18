"""Scoring a decode against what was actually sent."""


def levenshtein(a, b):
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def accuracy(truth, got):
    """Fraction of characters right, by edit distance. 1.0 is perfect."""
    truth = (truth or "").upper().strip()
    got = (got or "").upper().strip()
    if not truth:
        return 1.0 if not got else 0.0
    return max(0.0, 1.0 - levenshtein(truth, got) / len(truth))
