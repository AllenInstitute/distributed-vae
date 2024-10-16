import operator
from typing import Callable, Tuple, Any

ParserP = Callable[[str], Tuple[Any, str]]


class ParserError(Exception):
    def __init__(self, msg, content):
        super().__init__(f"{msg}: {content}")


def parse(p: ParserP, s: str) -> Tuple[Any, str]:
    (a, s) = p(s)
    return (a, s)


def sat(pred: Callable[["char"], bool]) -> ParserP:
    def fn(s: str) -> Tuple[str, str]:
        if not s:
            raise ParserError("Empty string", "")
        if pred(s[0]):
            return (s[0], s[1:])
        raise ParserError("Unexpected condition", s)

    return fn


def anyChar() -> ParserP:
    def fn(s: str) -> Tuple[str, str]:
        return (s[0], s[1:])

    return fn


def oneChar(c) -> ParserP:
    return sat(lambda c1: c == c1)


def anyDigit() -> ParserP:
    return sat(lambda c: c.isdigit())


def compose(p1: ParserP, p2: ParserP) -> ParserP:
    def fn(s: str) -> Tuple[Tuple[Any, Any], str]:
        (a, s1) = p1(s)
        (b, s2) = p2(s1)
        return ((a, b), s2)

    return fn


def choice(p1: ParserP, p2: ParserP) -> ParserP:
    def fn(s: str) -> Tuple[Any, str]:
        try:
            return p1(s)
        except ParserError:
            return p2(s)

    return fn
