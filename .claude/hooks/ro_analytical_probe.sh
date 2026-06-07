#!/bin/bash
# PreToolUse(Bash): cheap-first reminder before a compute-heavy batch. NON-BLOCKING —
# injects additionalContext the model sees (PLAN §8 analytical-probe; CLAUDE.md
# analytical-before-computational). Fires only on batch-launch command signatures.
input=$(cat)
cwd=$(echo "$input" | jq -r '.cwd // empty')
[[ "$cwd" != *lcas_v1.0* ]] && exit 0
cmd=$(echo "$input" | jq -r '.tool_input.command // empty')
# git operations are never batch launches — commit message bodies legitimately mention
# Pool(...), --seeds, sXXX, etc., and would otherwise trip the signature below.
echo "$cmd" | grep -qE '(^|[&;|])[[:space:]]*git[[:space:]]' && exit 0
# Batch-launch signatures used in this project (tight, to avoid firing on benign cmds).
if echo "$cmd" | grep -qE 'experiments/s[0-9]+[a-z]*[^ ]*\.py|run_[a-z_]+\.sh|Pool\(|--seeds|n_seeds|sbatch|S1[0-9][0-9]_'; then
  jq -n '{hookSpecificOutput:{hookEventName:"PreToolUse",additionalContext:"analytical-before-computational (CLAUDE.md): before this batch, state what a cheap re-score of existing checkpoints / result.json would answer, and show it insufficient. Name >=1 seed whose outcome is genuinely uncertain and run ONLY uncertain seeds. Save NPZ/JSON checkpoints; kill at 2x expected time."}}'
fi
exit 0
