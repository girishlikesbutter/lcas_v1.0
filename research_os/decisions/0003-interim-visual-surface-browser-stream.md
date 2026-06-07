# 0003 — Interim visual surface = local browser plot-stream

Status: accepted — 2026-06-04

Visual validation is first-class for this operator (eyes are a validation gate, PLAN §8).
Rather than wait for the Phase-4 web app, the interim surface is a local **browser
plot-stream**: PNGs in `research_os/render/stream/`, a static `index.html` that polls
`manifest.json` and renders newest-at-bottom, served by `python -m http.server` and
opened with `xdg-open`. `stream_add.py` is the "emit" primitive.

Why: true inline images are impossible in the real stack — Claude Code's TUI is
text-only, and tmux blocks Ghostty's image protocol. The browser stream delivers
show-on-the-fly + store + organise for ~zero build cost, and is throwaway (delete it,
lose only the cache — consistent with Q1).

Alternatives rejected: (a) Phase-4 web app now — too big, premature; (b) tmux + chafa
block-art pane — low fidelity, no store/organise.

Consequence: the remaining work is the dynamic-viz skill (decides *what* to draw, calls
`stream_add`) — the display itself is solved.
