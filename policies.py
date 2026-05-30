"""Reference policies: perfect counter vs flat-bet basic strategy."""

from __future__ import annotations

import json
from typing import Callable

from env import TABLE_MIN, BlackjackEnv
from strategy import optimal_action, optimal_bet

PolicyFn = Callable[[BlackjackEnv], dict]


def _action_payload(
    env: BlackjackEnv,
    *,
    bet_size: int,
    play: str,
    running: float,
    true_count: float,
) -> dict:
    return {
        "bet_size": bet_size,
        "action": play,
        "running_count": running,
        "true_count": round(true_count, 2),
    }


def smart_counter_policy() -> PolicyFn:
    def choose(env: BlackjackEnv) -> dict:
        tc = env._true_count()
        run = float(env._running)
        if env._phase == "bet":
            return _action_payload(
                env,
                bet_size=optimal_bet(TABLE_MIN, tc),
                play="none",
                running=run,
                true_count=tc,
            )
        if env._phase == "insurance":
            play = optimal_action(
                env._active_cards(),
                env._dealer[0],
                tc,
                can_double=False,
                can_split=False,
                insurance_offered=True,
            )
            return _action_payload(env, bet_size=0, play=play, running=run, true_count=tc)
        cards = env._active_cards()
        can_double = len(cards) == 2 and not env._doubled[env._active]
        can_split = (
            len(cards) == 2
            and cards[0] == cards[1]
            and len(env._player_hands) < 4
            and not env._doubled[env._active]
        )
        play = optimal_action(
            cards,
            env._dealer[0],
            tc,
            can_double=can_double,
            can_split=can_split,
            insurance_offered=False,
        )
        return _action_payload(env, bet_size=env._current_bet, play=play, running=run, true_count=tc)

    return choose


def dumb_flat_policy() -> PolicyFn:
    """Min bets, wrong counts, basic strategy without TC deviations."""

    from strategy import basic_action

    def choose(env: BlackjackEnv) -> dict:
        if env._phase == "bet":
            return _action_payload(
                env,
                bet_size=TABLE_MIN,
                play="none",
                running=0.0,
                true_count=0.0,
            )
        if env._phase == "insurance":
            return _action_payload(env, bet_size=0, play="none", running=0.0, true_count=0.0)
        cards = env._active_cards()
        play = basic_action(
            cards,
            env._dealer[0],
            can_double=len(cards) == 2 and not env._doubled[env._active],
            can_split=len(cards) == 2 and cards[0] == cards[1],
        )
        return _action_payload(env, bet_size=env._current_bet, play=play, running=0.0, true_count=0.0)

    return choose


def run_episode(env: BlackjackEnv, policy: PolicyFn, seed: int) -> float:
    env.reset(seed=seed)
    total = 0.0
    while True:
        result = env.step(policy(env))
        total += result.reward
        if result.terminated or result.truncated:
            break
    return total
