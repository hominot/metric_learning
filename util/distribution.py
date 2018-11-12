from itertools import product
from functools import reduce


def factor_expansion(exponents):
    ret = []
    for expanded in product(*[[0, exponent] for exponent in exponents]):
        ret.append(sum(expanded))
    signs = []
    for expanded in product(*[[1, -1] for _ in exponents]):
        signs.append(reduce(lambda x, y: x * y, expanded))
    return ret, signs


def integrate(exponents, signs):
    ret = 0.0
    for exponent, sign in zip(exponents, signs):
        ret += sign / (exponent + 1)
    return ret


def wallenius(x, w):
    w_chosen = [w[i] for i in x]
    d = sum(w) - sum(w_chosen)
    return integrate(*factor_expansion([weight / d for weight in w_chosen]))
