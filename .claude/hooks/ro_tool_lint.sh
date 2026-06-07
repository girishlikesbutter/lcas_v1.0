#!/bin/bash
# PostToolUse(Write|Edit): Tool drift-check (ADR-0007 §4). When an edit touches code a Tool
# card *reflects* — the substrate card itself, or a src/lib function its entry_point binds —
# re-ground the affected Tool(s) against the live source and surface drift (hash_moved /
# missing) the moment it appears. The dangerous case is the SILENT code-bump: you fix a
# bound function and the card's hash quietly goes stale.
#
# Advisory by design (soft, exit 0): the edit is legitimate; the HARD refusal lives at
# execution time (run_tool.py refuses to run a drifted Tool). This is the early warning so
# the next run isn't a surprise — repoint the entry_point + bump current_version/current_hash
# to clear it. (To make drift block-and-notify instead, change the final `exit 0` to exit 2.)
input=$(cat)
cwd=$(echo "$input" | jq -r '.cwd // empty')
[[ "$cwd" != *lcas_v1.0* ]] && exit 0
path=$(echo "$input" | jq -r '.tool_input.file_path // empty')
case "$path" in
  *.py) ;;
  */research_os/substrate/*.json) ;;
  *) exit 0 ;;
esac
cd "${CLAUDE_PROJECT_DIR:-$cwd}" 2>/dev/null || exit 0
python3 research_os/loop/tool_lint.py --for-file "$path" 1>&2   # silent unless THIS edit drifted a Tool
exit 0
