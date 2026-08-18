"""What operators actually send.

CW has a vocabulary of a few hundred words. Abbreviations exist precisely
because they are sent thousands of times a day, so a decoder that knows them
has a strong prior: given ?NX there is really only one thing it can be.

Ordered roughly by how often each is heard, because when a pattern fits more
than one word, frequency is the tie-breaker that a human uses without noticing.
"""

# The common abbreviations, Q-codes and prosign expansions of everyday CW.
COMMON = """
CQ DE K AR SK BT KN AS BK R RR TU 73 88 QSL QRZ QTH QRM QRN QSB QRP QRQ QRS
QRT QRV QRX QSY QSO QRL NAME HR HW UR ES ABT AGN ANT BK CPY CUL CUZ DX FB FER
GA GE GM GN GUD HI HPE MNI NR NW OM OP PSE PWR RIG RST SIG SRI TNX TKS TX RX
VY WX WL WUD YL XYL DR TEMP FT DEG C F WPM KEY BUG PADDLE STRAIGHT
GOOD MORNING EVENING AFTERNOON NIGHT DAY WARM COLD SUNNY RAIN SNOW WIND CLOUDY
HOT COOL NICE FINE VERY MUCH THANKS THANK YOU FOR THE AND WITH HERE THERE
FROM THIS THAT HAVE BEEN WILL BACK AGAIN RIGHT LEFT FIRST TIME WORK SIGNAL
SOLID COPY READ LOUD CLEAR WEAK STRONG NOISE BAND ANTENNA DIPOLE VERTICAL BEAM
WATTS POWER RADIO STATION HOME PARK POTA SOTA CONTEST CALL CALLING LISTENING
AM PM UTC LOCAL YRS OLD RETIRED HAM LICENSE YEARS AGO NEW OLD
"""
WORDS = [w for w in COMMON.split() if w]
RANK = {w: i for i, w in enumerate(WORDS)}

# Net traffic. A directed net has its own vocabulary and its own Q-signals,
# and the net control operator is exactly the person who leaves a decoder
# running -- he is copying a roster of stations rather than one conversation,
# and cannot ask all of them to repeat.
NET = ["NCS", "NET", "QNI", "QND", "QNS", "QNZ", "QRU", "QTC", "QSP",
       "CHECK", "CHECKIN", "RELAY", "TRAFFIC", "ROSTER", "LIST", "STBY",
       "WID", "THRU", "CLR", "LATE", "EARLY"]

# Reports and other stock tokens that are not words but appear constantly.
STOCK = ["5NN", "599", "579", "559", "588", "449", "339", "229", "119"]
STOCK += NET
for _s in STOCK:
    RANK.setdefault(_s, len(RANK))
WORDS += STOCK

BY_LENGTH = {}
for _w in WORDS:
    BY_LENGTH.setdefault(len(_w), []).append(_w)


def frequency_rank(word):
    """Lower is more common. Unknown words sort last."""
    return RANK.get(word, len(RANK) + 1)
