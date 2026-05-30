"""Generate showcase/data/replay.json from a smart-counter episode."""

from __future__ import annotations

import json
from pathlib import Path

from env import BlackjackEnv
from policies import smart_counter_policy

OUT = Path("showcase/data/replay.json")


def main() -> None:
    env = BlackjackEnv()
    policy = smart_counter_policy()
    obs = env.reset(seed=42)
    turns: list[dict] = []
    step = 0
    while True:
        action = policy(env)
        result = env.step(action)
        step += 1
        turns.append(
            {
                "step": step,
                "observation": obs,
                "reasoning": (
                    f"TC={action['true_count']:.1f} bet={action['bet_size']} "
                    f"play={action['action']} (optimal bet {result.info.get('optimal_bet')})"
                ),
                "action": action,
                "reward": result.reward,
                "terminated": result.terminated,
                "truncated": result.truncated,
                "info": result.info,
            }
        )
        obs = result.observation
        if result.terminated or result.truncated:
            break

    ep_id = "sample-ep-1"
    payload = {
        "run": {"scores": {"mean_reward": sum(t["reward"] for t in turns)}},
        "episodes": [{"id": ep_id, "seed": 42, "total_reward": sum(t["reward"] for t in turns)}],
        "replay": {ep_id: turns},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path("showcase/data/replay.js").write_text(
        "window.REPLAY = " + json.dumps(payload) + ";\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({len(turns)} turns)")


if __name__ == "__main__":
    main()
