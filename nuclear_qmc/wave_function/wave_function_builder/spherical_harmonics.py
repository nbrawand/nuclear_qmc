def Y1m1(r):
    return r[1]


def Y11(r):
    return r[0]


def Y10(r):
    return r[2]


def Y00(r):
    return 1.0


SPHERICAL_HARMONICS = {f.__name__: f for f in [
    Y00
    , Y11
    , Y10
    , Y1m1
]}
