---
name: attitude-viz
description: "Generate 2x2 attitude comparison plots for any two states (q0 + omega). Compares two arbitrary states side by side with custom labels. Use when the user asks to visualise, compare, or plot attitudes/omegas. Triggers on: attitude viz, show me the attitude, plot the attitude, visualise seed, compare attitudes, compare candidates."
---

# Attitude Visualisation

Generate a 2x2 comparison plot of two states (attitude + omega) for one or more seeds. Each PNG shows:
- **Top-left / Top-right:** Satellite mesh + body axes rotated by each state's q0
- **Bottom-left / Bottom-right:** Satellite mesh in body frame + omega arrow for each state
- **All panels:** Observer direction vector (cyan)

## How to run

```bash
# Truth vs winner (default)
python3 notebooks/inversion/lib/attitude_viz.py 27

# Truth vs specific candidate
python3 notebooks/inversion/lib/attitude_viz.py 27 --b candidate:1

# Winner vs candidate with custom labels
python3 notebooks/inversion/lib/attitude_viz.py 27 --a winner --b candidate:1 \
    --label-a "Winner" --label-b "Rank #2" --prefix micro93

# Multiple seeds
python3 notebooks/inversion/lib/attitude_viz.py 6 24 36 --prefix micro93
```

## State specifications (--a, --b)

- `truth` — true state from trajectory database (default for --a)
- `winner` — winner from result.json (default for --b)
- `candidate:N` — hi-fi candidate N from result.json (0=winner, 1=rank#2, etc.)

## From Python

```python
from lib.attitude_viz import generate_attitude_viz
# Compare winner vs candidate #1
generate_attitude_viz(seeds=[27], prefix='micro93',
                      state_a='winner', state_b='candidate:1',
                      label_a='Winner', label_b='Rank #2')
```

## What to do

1. Parse the user's request for seed numbers, state specs, labels, and prefix.
2. Run the script with appropriate arguments.
3. Report the saved PNG path and display the image.
