"""Six-deck blackjack with Hi-Lo counting verification and multi-axis rewards."""

from __future__ import annotations

import json
import math
import random
from typing import Any

from bench_common.env_sdk.base import BaseEnv, StepResult

from strategy import hand_totals, optimal_action, optimal_bet

TABLE_MIN = 10
START_BANKROLL = 1000
MAX_STEPS = 35
DECKS = 6
SHUFFLE_AT = 78  # 75% penetration of 312 cards
PLAY_ACTIONS = frozenset({"hit", "stand", "double", "split", "insurance", "none"})


def hi_lo_delta(rank: int) -> int:
    if 2 <= rank <= 6:
        return 1
    if 7 <= rank <= 9:
        return 0
    return -1


def _build_shoe(rng: random.Random) -> list[int]:
    shoe: list[int] = []
    ranks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11]
    for _ in range(DECKS):
        shoe.extend(ranks * 4)
    rng.shuffle(shoe)
    return shoe


class BlackjackEnv(BaseEnv):
    def __init__(self) -> None:
        self._rng = random.Random(0)
        self._shoe: list[int] = []
        self._running = 0
        self._step = 0
        self._bankroll = START_BANKROLL
        self._phase = "bet"
        self._player_hands: list[list[int]] = [[]]
        self._hand_bets: list[int] = [0]
        self._active = 0
        self._doubled: list[bool] = [False]
        self._stood: list[bool] = [False]
        self._dealer: list[int] = []
        self._hole_hidden = True
        self._insurance_offered = False
        self._insurance_taken = False
        self._current_bet = 0
        self._hand_number = 0
        self._last_outcome = ""
        self._episode_profit = 0.0

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        if seed is None:
            seed = 0
        self._rng = random.Random(seed)
        self._shoe = _build_shoe(self._rng)
        self._running = 0
        self._step = 0
        self._bankroll = START_BANKROLL
        self._phase = "bet"
        self._player_hands = [[]]
        self._hand_bets = [0]
        self._active = 0
        self._doubled = [False]
        self._stood = [False]
        self._dealer = []
        self._hole_hidden = True
        self._insurance_offered = False
        self._insurance_taken = False
        self._current_bet = 0
        self._hand_number = 0
        self._last_outcome = "Episode started — place a bet."
        self._episode_profit = 0.0
        return self._observation()

    def step(self, action: Any) -> StepResult:
        payload = self._parse_action(action)
        bet_size = int(float(payload.get("bet_size", TABLE_MIN)))
        play = str(payload.get("action", "none")).strip().lower()
        agent_run = float(payload.get("running_count", 0))
        agent_true = float(payload.get("true_count", 0))

        reward = 0.0
        illegal = False
        optimal_play = "none"

        if self._phase == "bet":
            optimal_play = "none"
            opt_bet = optimal_bet(TABLE_MIN, self._true_count())
            reward += self._betting_reward(bet_size, opt_bet)
            reward += self._counting_reward(agent_run, agent_true)
            if bet_size < TABLE_MIN or bet_size > self._bankroll:
                reward -= 5.0
                illegal = True
                self._last_outcome = f"Illegal bet {bet_size}."
            else:
                self._start_hand(bet_size)
        elif self._phase == "insurance":
            optimal_play = optimal_action(
                self._active_cards(),
                self._dealer[0],
                self._true_count(),
                can_double=False,
                can_split=False,
                insurance_offered=True,
            )
            reward += self._counting_reward(agent_run, agent_true)
            reward += self._strategy_reward(play, optimal_play, illegal_move=False)
            if play == "insurance":
                cost = max(1, self._hand_bets[self._active] // 2)
                if cost <= self._bankroll:
                    self._bankroll -= cost
                    self._insurance_taken = True
                self._phase = "player"
            elif play in ("none", "stand"):
                self._phase = "player"
            else:
                reward -= 5.0
                illegal = True
        elif self._phase == "player":
            cards = self._active_cards()
            can_double = len(cards) == 2 and not self._doubled[self._active]
            can_split = (
                len(cards) == 2
                and cards[0] == cards[1]
                and len(self._player_hands) < 4
                and not self._doubled[self._active]
            )
            optimal_play = optimal_action(
                cards,
                self._dealer[0],
                self._true_count(),
                can_double=can_double,
                can_split=can_split,
                insurance_offered=False,
            )
            reward += self._counting_reward(agent_run, agent_true)
            if play not in PLAY_ACTIONS or play == "none":
                reward -= 5.0
                illegal = True
            elif not self._apply_player_action(play, can_double, can_split):
                reward -= 5.0
                illegal = True
            else:
                reward += self._strategy_reward(play, optimal_play, illegal_move=False)

        terminated = False
        truncated = False
        self._step += 1

        if self._bankroll < TABLE_MIN:
            terminated = True
            self._last_outcome += " Busted bankroll."
        elif self._step >= MAX_STEPS:
            truncated = True
            self._last_outcome += f" Step cap ({MAX_STEPS})."

        return StepResult(
            observation=self._observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=self._info(agent_run, agent_true, optimal_play, bet_size),
        )

    def _parse_action(self, action: Any) -> dict[str, Any]:
        if isinstance(action, dict):
            return action
        if isinstance(action, str):
            text = action.strip()
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
            return {"action": text, "bet_size": TABLE_MIN, "running_count": 0, "true_count": 0}
        return {"action": "none", "bet_size": TABLE_MIN, "running_count": 0, "true_count": 0}

    def _cards_remaining(self) -> int:
        return len(self._shoe)

    def _decks_remaining(self) -> float:
        return max(0.5, self._cards_remaining() / 52.0)

    def _true_count(self) -> float:
        return self._running / self._decks_remaining()

    def _count_visible(self, rank: int) -> None:
        self._running += hi_lo_delta(rank)

    def _maybe_shuffle(self) -> None:
        if self._cards_remaining() <= SHUFFLE_AT:
            self._shoe = _build_shoe(self._rng)
            self._running = 0
            self._last_outcome = "Shoe shuffled at 75% penetration."

    def _draw(self, *, visible: bool = True) -> int:
        if not self._shoe:
            self._shoe = _build_shoe(self._rng)
        card = self._shoe.pop()
        if visible:
            self._count_visible(card)
        return card

    def _start_hand(self, bet_size: int) -> None:
        self._maybe_shuffle()
        self._hand_number += 1
        self._current_bet = bet_size
        self._bankroll -= bet_size
        self._player_hands = [[]]
        self._hand_bets = [bet_size]
        self._active = 0
        self._doubled = [False]
        self._stood = [False]
        self._dealer = []
        self._hole_hidden = True
        self._insurance_offered = False
        self._insurance_taken = False

        for _ in range(2):
            self._player_hands[0].append(self._draw(visible=True))
        self._dealer.append(self._draw(visible=True))
        self._dealer.append(self._draw(visible=False))

        p_total, _ = hand_totals(self._player_hands[0])
        d_total, soft = hand_totals(self._dealer)

        if self._dealer[0] == 11:
            self._insurance_offered = True
            self._phase = "insurance"
            self._last_outcome = "Dealer shows Ace — insurance?"
            return

        if p_total == 21 and len(self._player_hands[0]) == 2:
            self._resolve_natural(player_blackjack=True)
            return
        if d_total == 21 and not soft and len(self._dealer) == 2:
            self._resolve_natural(player_blackjack=False)
            return

        self._phase = "player"
        self._last_outcome = "Player's turn."

    def _active_cards(self) -> list[int]:
        return self._player_hands[self._active]

    def _apply_player_action(self, play: str, can_double: bool, can_split: bool) -> bool:
        if play == "split":
            if not can_split:
                return False
            cards = self._active_cards()
            self._player_hands[self._active] = [cards[0]]
            self._player_hands.insert(self._active + 1, [cards[1]])
            self._hand_bets.insert(self._active + 1, self._hand_bets[self._active])
            self._doubled.insert(self._active + 1, False)
            self._stood.insert(self._active + 1, False)
            if self._bankroll < self._hand_bets[self._active]:
                return False
            self._bankroll -= self._hand_bets[self._active]
            self._player_hands[self._active].append(self._draw())
            self._player_hands[self._active + 1].append(self._draw())
            return True

        if play == "double":
            if not can_double:
                return False
            if self._bankroll < self._hand_bets[self._active]:
                return False
            self._bankroll -= self._hand_bets[self._active]
            self._hand_bets[self._active] *= 2
            self._doubled[self._active] = True
            self._player_hands[self._active].append(self._draw())
            self._stood[self._active] = True
            return self._advance_after_player_action()

        if play == "hit":
            self._player_hands[self._active].append(self._draw())
            total, _ = hand_totals(self._active_cards())
            if total > 21:
                self._stood[self._active] = True
                return self._advance_after_player_action()
            return True

        if play == "stand":
            self._stood[self._active] = True
            return self._advance_after_player_action()

        return False

    def _advance_after_player_action(self) -> bool:
        total, _ = hand_totals(self._active_cards())
        if total > 21:
            pass
        if self._active < len(self._player_hands) - 1:
            self._active += 1
            if not self._stood[self._active]:
                return True
            return self._advance_after_player_action()
        self._dealer_play()
        self._resolve_hand()
        return True

    def _dealer_play(self) -> None:
        self._hole_hidden = False
        if len(self._dealer) > 1:
            self._count_visible(self._dealer[1])
        while True:
            total, soft = hand_totals(self._dealer)
            if total < 17:
                self._dealer.append(self._draw())
                continue
            if total == 17 and soft:
                self._dealer.append(self._draw())
                continue
            break

    def _resolve_natural(self, *, player_blackjack: bool) -> None:
        self._hole_hidden = False
        if len(self._dealer) > 1:
            self._count_visible(self._dealer[1])
        if player_blackjack:
            win = int(self._hand_bets[0] * 2.5)
            self._bankroll += win
            self._last_outcome = "Player blackjack — pays 3:2."
        else:
            self._last_outcome = "Dealer blackjack."
        self._phase = "bet"
        self._player_hands = [[]]
        self._dealer = []

    def _resolve_hand(self) -> None:
        self._hole_hidden = False
        d_total, _ = hand_totals(self._dealer)
        results: list[str] = []
        net = 0
        for i, cards in enumerate(self._player_hands):
            bet = self._hand_bets[i]
            p_total, _ = hand_totals(cards)
            if p_total > 21:
                results.append(f"Hand {i+1}: bust (-{bet})")
                net -= bet
                continue
            if d_total > 21 or p_total > d_total:
                self._bankroll += bet * 2
                net += bet
                results.append(f"Hand {i+1}: win (+{bet})")
            elif p_total == d_total:
                self._bankroll += bet
                results.append(f"Hand {i+1}: push")
            else:
                net -= bet
                results.append(f"Hand {i+1}: loss (-{bet})")

        if self._insurance_taken and self._dealer[0] == 11 and len(self._dealer) >= 2:
            if hand_totals(self._dealer)[0] == 21:
                ins_bet = max(1, self._hand_bets[0] // 2)
                if hand_totals(self._dealer)[0] == 21:
                    self._bankroll += ins_bet * 3
                    net += ins_bet * 2

        self._episode_profit += net
        self._last_outcome = "; ".join(results)
        self._phase = "bet"
        self._player_hands = [[]]
        self._dealer = []
        self._insurance_offered = False
        self._insurance_taken = False

    def _betting_reward(self, chosen: int, optimal: int) -> float:
        if optimal <= 0:
            return 0.0
        err = abs(chosen - optimal) / optimal
        return max(0.0, 1.0 - err)

    def _counting_reward(self, agent_run: float, agent_true: float) -> float:
        drift_run = abs(agent_run - self._running)
        drift_true = abs(agent_true - self._true_count())
        return -0.05 * drift_run - 0.1 * drift_true

    def _strategy_reward(self, chosen: str, optimal: str, *, illegal_move: bool) -> float:
        if illegal_move:
            return -5.0
        if chosen == optimal:
            return 1.0
        if chosen in PLAY_ACTIONS and optimal in PLAY_ACTIONS:
            return -0.5
        return -0.5

    def _observation(self) -> dict[str, Any]:
        cards = self._active_cards() if self._phase == "player" else []
        total, soft = hand_totals(cards) if cards else (0, False)
        return {
            "phase": self._phase,
            "bankroll": self._bankroll,
            "table_min": TABLE_MIN,
            "cards_remaining": self._cards_remaining(),
            "hand_number": self._hand_number,
            "step": self._step,
            "max_steps": MAX_STEPS,
            "player_hands": self._player_hands,
            "active_hand": self._active,
            "hand_total": total,
            "hand_soft": soft,
            "dealer_upcard": self._dealer[0] if self._dealer else None,
            "dealer_hole_hidden": self._hole_hidden,
            "current_bet": self._current_bet,
            "insurance_offered": self._insurance_offered,
            "can_double": self._phase == "player"
            and len(cards) == 2
            and not self._doubled[self._active],
            "can_split": self._phase == "player"
            and len(cards) == 2
            and cards[0] == cards[1]
            and len(self._player_hands) < 4,
            "valid_actions": self._valid_actions(),
            "message": self._last_outcome,
            "ground_truth_running_count": self._running,
            "ground_truth_true_count": round(self._true_count(), 2),
            "optimal_bet_hint": optimal_bet(TABLE_MIN, self._true_count()),
        }

    def _valid_actions(self) -> list[str]:
        if self._phase == "bet":
            return ["none"]
        if self._phase == "insurance":
            return ["insurance", "none"]
        if self._phase == "player":
            acts = ["hit", "stand"]
            cards = self._active_cards()
            if len(cards) == 2 and not self._doubled[self._active]:
                acts.append("double")
            if (
                len(cards) == 2
                and cards[0] == cards[1]
                and len(self._player_hands) < 4
            ):
                acts.append("split")
            return acts
        return []

    def _info(
        self,
        agent_run: float,
        agent_true: float,
        optimal_play: str,
        bet_size: int,
    ) -> dict[str, str]:
        hole = None if self._hole_hidden else (self._dealer[1] if len(self._dealer) > 1 else None)
        return {
            "phase": self._phase,
            "bankroll": str(self._bankroll),
            "player_hands": json.dumps(self._player_hands),
            "dealer_upcard": str(self._dealer[0] if self._dealer else ""),
            "dealer_hole": str(hole if hole is not None else "hidden"),
            "dealer_hole_hidden": str(self._hole_hidden).lower(),
            "ground_truth_running": str(self._running),
            "ground_truth_true": str(round(self._true_count(), 2)),
            "agent_running": str(agent_run),
            "agent_true": str(agent_true),
            "optimal_bet": str(optimal_bet(TABLE_MIN, self._true_count())),
            "chosen_bet": str(bet_size),
            "optimal_action": optimal_play,
            "cards_remaining": str(self._cards_remaining()),
            "shoe_remaining": json.dumps(self._shoe[-40:]),  # tail for showcase hover
            "outcome": self._last_outcome,
            "hand_number": str(self._hand_number),
            "step": str(self._step),
            "episode_profit": str(round(self._episode_profit, 2)),
            "profitable": str(self._bankroll > START_BANKROLL).lower(),
        }
