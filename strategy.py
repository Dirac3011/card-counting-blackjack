"""Basic strategy + Illustrious 18 true-count deviations."""

from __future__ import annotations

import math
from typing import Iterable

_UP = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11)

_HARD: dict[tuple[int, int], str] = {}
for d in _UP:
    for t in range(5, 12):
        _HARD[(t, d)] = "H"
    _HARD[(12, 2)] = _HARD[(12, 3)] = "H"
    _HARD[(12, 4)] = _HARD[(12, 5)] = _HARD[(12, 6)] = "H"
    for dd in (7, 8, 9, 10, 11):
        _HARD[(12, dd)] = "H"
    _HARD[(13, 2)] = "S"
    for dd in (3, 4, 5, 6):
        _HARD[(13, dd)] = "S"
    for dd in (7, 8, 9, 10, 11):
        _HARD[(13, dd)] = "H"
    for t in range(14, 18):
        for dd in (2, 3, 4, 5, 6):
            _HARD[(t, dd)] = "S"
        for dd in (7, 8, 9, 10, 11):
            _HARD[(t, dd)] = "H"
    for t in range(17, 22):
        for dd in _UP:
            _HARD[(t, dd)] = "S"
    _HARD[(8, 5)] = _HARD[(8, 6)] = "D"
    for dd in (3, 4, 5, 6):
        _HARD[(9, dd)] = "D"
    for dd in (2, 3, 4, 5, 6):
        _HARD[(10, dd)] = _HARD[(11, dd)] = "D"
    for dd in (7, 8, 9, 10, 11):
        _HARD[(9, dd)] = _HARD[(10, dd)] = _HARD[(11, dd)] = "H"

_SOFT: dict[tuple[int, int], str] = {}
for d in _UP:
    _SOFT[(20, d)] = _SOFT[(19, d)] = "S"
    _SOFT[(18, d)] = "S" if d in (2, 3, 4, 5, 6) else "H"
    _SOFT[(17, d)] = "S" if d in (2, 3, 4, 5, 6) else "H"
    for t in (13, 14, 15, 16):
        _SOFT[(t, d)] = "H" if d in (2, 3, 4) else ("S" if d in (5, 6) else "H")

_PAIR: dict[tuple[int, int], str] = {}
for d in _UP:
    _PAIR[(11, d)] = "P"
    _PAIR[(8, d)] = "P"
    _PAIR[(2, d)] = _PAIR[(3, d)] = _PAIR[(7, d)] = "P"
    _PAIR[(6, 2)] = _PAIR[(6, 3)] = _PAIR[(6, 4)] = _PAIR[(6, 5)] = _PAIR[(6, 6)] = "P"
    _PAIR[(4, 5)] = _PAIR[(4, 6)] = "P"
    _PAIR[(9, d)] = "P" if d not in (7, 10, 11) else "S"
    _PAIR[(5, d)] = "D" if d <= 9 else "H"
    _PAIR[(10, d)] = "S"

_DEVIATIONS: list[tuple[str, int, int, float, str]] = [
    ("hard", 16, 10, 0.0, "stand"),
    ("hard", 15, 10, 4.0, "stand"),
    ("hard", 12, 2, 3.0, "stand"),
    ("hard", 12, 3, 2.0, "stand"),
    ("hard", 12, 4, 0.0, "stand"),
    ("hard", 12, 5, -2.0, "stand"),
    ("hard", 12, 6, -1.0, "stand"),
    ("hard", 13, 2, -1.0, "stand"),
    ("hard", 13, 3, -2.0, "stand"),
    ("hard", 16, 9, 4.0, "stand"),
    ("hard", 10, 10, 4.0, "double"),
    ("hard", 11, 11, 1.0, "double"),
    ("hard", 9, 2, 1.0, "double"),
    ("hard", 9, 7, 3.0, "double"),
    ("pair", 10, 5, 5.0, "split"),
    ("pair", 10, 6, 4.0, "split"),
]

_ACTION_MAP = {"H": "hit", "S": "stand", "D": "double", "P": "split"}


def card_value(rank: int) -> int:
    if rank == 11:
        return 11
    if rank >= 10:
        return 10
    return rank


def hand_totals(cards: Iterable[int]) -> tuple[int, bool]:
    cards = list(cards)
    total = 0
    aces = 0
    for r in cards:
        if r == 11:
            aces += 1
            total += 1
        elif r >= 10:
            total += 10
        else:
            total += r
    soft = False
    if aces and total + 10 <= 21:
        total += 10
        soft = True
    return total, soft


def dealer_up_rank(upcard: int) -> int:
    return 11 if upcard == 11 else min(upcard, 10)


def basic_action(
    cards: list[int],
    dealer_up: int,
    *,
    can_double: bool,
    can_split: bool,
) -> str:
    if len(cards) == 2 and can_split and cards[0] == cards[1]:
        key = (card_value(cards[0]), dealer_up_rank(dealer_up))
        act = _PAIR.get(key, "S")
        return _ACTION_MAP.get(act, "stand")
    total, soft = hand_totals(cards)
    d = dealer_up_rank(dealer_up)
    if soft:
        act = _SOFT.get((total, d), "H")
    else:
        act = _HARD.get((total, d), "S" if total >= 17 else "H")
    mapped = _ACTION_MAP.get(act, "hit")
    if mapped == "double" and not can_double:
        mapped = "hit"
    if mapped == "split" and not can_split:
        mapped = "hit"
    return mapped


def optimal_action(
    cards: list[int],
    dealer_up: int,
    true_count: float,
    *,
    can_double: bool,
    can_split: bool,
    insurance_offered: bool,
) -> str:
    if insurance_offered:
        return "insurance" if true_count >= 3.0 else "none"

    total, soft = hand_totals(cards)
    d = dealer_up_rank(dealer_up)

    if len(cards) == 2 and can_split and cards[0] == cards[1]:
        pair_val = card_value(cards[0])
        for tag, ptot, dup, tc, act in _DEVIATIONS:
            if tag == "pair" and pair_val == ptot and d == dup and true_count >= tc:
                if act == "split" and can_split:
                    return "split"
        return basic_action(cards, dealer_up, can_double=can_double, can_split=can_split)

    if not soft:
        for tag, ptot, dup, tc, act in _DEVIATIONS:
            if tag == "hard" and total == ptot and d == dup and true_count >= tc:
                if act == "double" and can_double:
                    return "double"
                return act

    return basic_action(cards, dealer_up, can_double=can_double, can_split=can_split)


def optimal_bet(table_min: int, true_count: float) -> int:
    if true_count < 1.0:
        return table_min
    return int(table_min * max(1, math.floor(true_count)))
