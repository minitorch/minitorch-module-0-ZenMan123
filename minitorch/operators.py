"""Collection of the core mathematical operators used throughout the code base."""

import math

from typing import Callable, Iterable, List


def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> bool:
    return x < y


def eq(x: float, y: float) -> bool:
    return x == y


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def relu(x: float) -> float:
    return x if x > 0 else 0.0


def log(x: float) -> float:
    return math.log(x)


def exp(x: float) -> float:
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    return d / x


def inv(x: float) -> float:
    return 1.0 / x


def inv_back(x: float, d: float) -> float:
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0.0


def map(fn: Callable[[float], float], it: Iterable[float]) -> Iterable[float]:
    for x in it:
        yield fn(x)


def zipWith(
    fn: Callable[[float, float], float],
    it_left: Iterable[float],
    it_right: Iterable[float],
) -> Iterable[float]:
    for x, y in zip(it_left, it_right):
        yield fn(x, y)


def reduce(fn: Callable[[float, float], float], it: Iterable[float], start: float) -> float:
    result = start
    for value in it:
        result = fn(result, value)
    return result


def negList(lst: List[float]) -> List[float]:
    return list(map(neg, lst))


def addLists(left_lst: List[float], right_lst: List[float]) -> List[float]:
    return list(zipWith(add, left_lst, right_lst))


def sum(lst: List[float]) -> float:
    return reduce(add, lst, 0.0)


def prod(lst: List[float]) -> float:
    return reduce(mul, lst, 1.0)
