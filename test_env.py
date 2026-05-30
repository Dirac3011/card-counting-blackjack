"""Determinism and policy discrimination checks."""

from __future__ import annotations

import json

from env import BlackjackEnv
from policies import dumb_flat_policy, run_episode, smart_counter_policy


def _replay(seed: int, actions: list[dict]) -> list[dict[str, str]]:
    env = BlackjackEnv()
    env.reset(seed=seed)
    traces: list[dict[str, str]] = []
    for act in actions:
        res = env.step(act)
        traces.append(res.info)
        if res.terminated or res.truncated:
            break
    return traces


def _record_episode(seed: int, max_actions: int = 25) -> list[dict]:
    env = BlackjackEnv()
    policy = smart_counter_policy()
    env.reset(seed=seed)
    script: list[dict] = []
    for _ in range(max_actions):
        act = policy(env)
        script.append(act)
        res = env.step(act)
        if res.terminated or res.truncated:
            break
    return script


def test_determinism() -> None:
    for seed in (0, 1, 42, 99):
        script = _record_episode(seed)
        assert _replay(seed, script) == _replay(seed, script), f"seed {seed}"
    print("determinism: OK")


def test_policies_differ() -> None:
    seeds = list(range(6))
    smart = smart_counter_policy()
    dumb = dumb_flat_policy()
    r_smart = sum(run_episode(BlackjackEnv(), smart, s) for s in seeds) / len(seeds)
    r_dumb = sum(run_episode(BlackjackEnv(), dumb, s) for s in seeds) / len(seeds)
    print(f"mean reward — smart: {r_smart:.3f}, dumb: {r_dumb:.3f}")
    assert r_smart > r_dumb + 2.0
    print("policy discrimination: OK")


def test_action_json_roundtrip() -> None:
    env = BlackjackEnv()
    env.reset(seed=1)
    payload = {
        "bet_size": 10,
        "action": "none",
        "running_count": 0,
        "true_count": 0.0,
    }
    res = env.step(json.dumps(payload))
    assert res.reward > -5
    print("json action: OK")


if __name__ == "__main__":
    test_action_json_roundtrip()
    test_determinism()
    test_policies_differ()
    print("All tests passed.")
