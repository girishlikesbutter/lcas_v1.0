#!/bin/bash
# PostToolUse(Write|Edit): hard epistemic gate on a written run_record / claim_card
# (N-scope, oracle-coherent). Soft heuristics are audit-only (gate_check.py --all),
# kept OUT of the per-write hook to stay non-noisy. exit 2 -> stderr fed to the model.
input=$(cat)
cwd=$(echo "$input" | jq -r '.cwd // empty')
[[ "$cwd" != *lcas_v1.0* ]] && exit 0
path=$(echo "$input" | jq -r '.tool_input.file_path // empty')
case "$path" in
  *"/research_os/records/"*.json | *"/research_os/claims/"*.json) ;;
  *) exit 0 ;;
esac
cd "${CLAUDE_PROJECT_DIR:-$cwd}" 2>/dev/null || exit 0
python3 research_os/loop/gate_check.py "$path"   # hard-only; exit 2 on violation
exit $?
