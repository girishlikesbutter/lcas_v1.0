#!/bin/bash
# PostToolUse(Write|Edit): when a substrate component file changes, recompute the
# blast radius and flip dependent live/draft claims to needs_replication (PLAN §3, Q6).
# Safe-direction mutation only (never asserts truth) -> may act automatically.
input=$(cat)
cwd=$(echo "$input" | jq -r '.cwd // empty')
[[ "$cwd" != *lcas_v1.0* ]] && exit 0
path=$(echo "$input" | jq -r '.tool_input.file_path // empty')
[[ "$path" != *"/research_os/substrate/"*".json" ]] && exit 0
cd "${CLAUDE_PROJECT_DIR:-$cwd}" 2>/dev/null || exit 0
# Report (and flips) go to stderr so they surface in the hook output.
python3 research_os/loop/blast_radius.py 1>&2
exit 0
