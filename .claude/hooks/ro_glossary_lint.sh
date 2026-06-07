#!/bin/bash
# PostToolUse(Write|Edit): lint a survey writeup OR a store object against the canon
# glossary. Covers both authoring surfaces — experiment prose AND the structured store
# (goals/records/claims/pipelines/contracts), so new drift is caught wherever it lands.
# The linter is JSON-aware: store .json files lint only their prose fields, never IDs
# or synonyms_blocked declarations. Soft (exit 0); hits go to stderr once canon exists.
input=$(cat)
cwd=$(echo "$input" | jq -r '.cwd // empty')
[[ "$cwd" != *lcas_v1.0* ]] && exit 0
path=$(echo "$input" | jq -r '.tool_input.file_path // empty')
case "$path" in
  */survey/experiments/*.md) ;;
  */research_os/goals/*.json|*/research_os/records/*.json|*/research_os/claims/*.json|*/research_os/pipelines/*.json|*/research_os/contracts/*.json) ;;
  *) exit 0 ;;
esac
cd "${CLAUDE_PROJECT_DIR:-$cwd}" 2>/dev/null || exit 0
python3 research_os/loop/glossary_lint.py "$path" 1>&2
exit 0
