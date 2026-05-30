# Card Counting Blackjack

Mesocosm environment: **6-deck shoe**, **Hi-Lo** counting, **Illustrious 18** deviations, and dynamic bet sizing.

| | |
|---|---|
| **ID** | `card-counting-blackjack` |
| **Model** | `openai/gpt-4o` |
| **Max steps** | 35 per episode |

## Action (JSON each step)

```json
{
  "bet_size": 10,
  "action": "hit",
  "running_count": 2,
  "true_count": 1.5
}
```

Actions: `hit`, `stand`, `double`, `split`, `insurance`, `none` (betting phase).

## Local test

```powershell
cd card-counting-blackjack
..\.venv\Scripts\python.exe test_env.py
..\.venv\Scripts\mesocosm.exe validate benchanything.json
python adapter.py
```

## Platform

```powershell
mesocosm auth login
mesocosm env submit --name "Card Counting Blackjack" --github-url https://github.com/<user>/card-counting-blackjack --solo
mesocosm run create --domain <DOMAIN_ID> --vow-version 1.0.0 --model openai/gpt-4o --episodes 8 --visibility gallery_public --solo
mesocosm run export <RUN_ID> -o showcase/data/replay.json
python gen_sample_replay.py
```

Showcase: `https://<user>.github.io/card-counting-blackjack/`
