#!/usr/bin/env python3
"""Multiple persistent browser terminals (PLAN §5/§11, Q11).

"Focused work stays in-terminal" — Girish wants real Claude Code sessions in the
browser, plural and switchable, not a GUI that replaces the CLI. This is the server
side: a ``TerminalManager`` that owns a set of long-lived PTYs (each an interactive
login ``bash`` in the repo root; the operator types ``claude`` inside one to start a
session). Sessions persist across browser reloads — output is buffered in a ring so a
re-attaching client is caught up — and several browser tabs can attach to the same
PTY at once (shared view, like tmux).

Pure stdlib PTY (``pty``/``fcntl``/``termios``) + asyncio ``add_reader``; no node-pty,
no ttyd. One process, one stack.
"""
from __future__ import annotations

import asyncio
import fcntl
import os
import pty
import signal
import struct
import termios
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(  # …/lcas_v1.0
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
SCROLLBACK_BYTES = 256 * 1024  # ~256 KB caught-up buffer per terminal


@dataclass
class Terminal:
    id: str
    title: str
    fd: int
    pid: int
    cols: int = 80
    rows: int = 24
    buffer: bytearray = field(default_factory=bytearray)
    clients: set = field(default_factory=set)  # connected asyncio.Queue sinks
    created_at: float = 0.0
    _reader_installed: bool = False


class TerminalManager:
    """Owns the live PTYs. Spawned lazily; reaped on close."""

    def __init__(self, shell: str | None = None, cwd: str = REPO_ROOT):
        self.shell = shell or os.environ.get("SHELL", "/bin/bash")
        self.cwd = cwd
        self.terms: dict[str, Terminal] = {}
        self._seq = 0
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- lifecycle ---------------------------------------------------------
    def _spawn(self, term: Terminal):
        pid, fd = pty.fork()
        if pid == 0:  # child
            os.chdir(self.cwd)
            env = os.environ.copy()
            env.update({
                "TERM": "xterm-256color",
                "COLORTERM": "truecolor",
                "RESEARCH_OS_TERMINAL": term.id,
                "LINES": str(term.rows), "COLUMNS": str(term.cols),
            })
            os.execvpe(self.shell, [self.shell, "-l"], env)
            os._exit(127)  # unreachable
        # parent
        term.fd, term.pid = fd, pid
        self._set_winsize(fd, term.rows, term.cols)
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    @staticmethod
    def _set_winsize(fd: int, rows: int, cols: int):
        try:
            fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        except OSError:
            pass

    def create(self, title: str = "", cols: int = 80, rows: int = 24) -> Terminal:
        import time
        self._seq += 1
        tid = f"t{self._seq}"
        term = Terminal(id=tid, title=title or f"terminal {self._seq}",
                        cols=cols, rows=rows, fd=-1, pid=-1, created_at=time.time())
        self._spawn(term)
        self.terms[tid] = term
        return term

    def list(self) -> list[dict]:
        out = []
        for t in self.terms.values():
            out.append({
                "id": t.id, "title": t.title, "cols": t.cols, "rows": t.rows,
                "clients": len(t.clients), "alive": self._alive(t),
                "created_at": t.created_at,
            })
        return out

    def _alive(self, t: Terminal) -> bool:
        try:
            pid, _ = os.waitpid(t.pid, os.WNOHANG)
            return pid == 0
        except (ChildProcessError, OSError):
            return False

    def close(self, tid: str):
        t = self.terms.pop(tid, None)
        if not t:
            return
        if self._loop:
            try:
                self._loop.remove_reader(t.fd)
            except Exception:
                pass
        for q in list(t.clients):
            try:
                q.put_nowait(None)  # signal disconnect
            except Exception:
                pass
        try:
            os.kill(t.pid, signal.SIGHUP)
        except OSError:
            pass
        try:
            os.close(t.fd)
        except OSError:
            pass
        # Reap the child. SIGHUP-then-close doesn't terminate it synchronously, so a
        # single waitpid here would usually return 0 and leave a zombie until server
        # exit (the EOF-driven _reap never fires once we've removed the reader). Poll
        # waitpid(WNOHANG) a few times off the event loop instead — non-blocking.
        self._reap_pid(t.pid)

    def _reap_pid(self, pid: int, attempts: int = 12):
        """Non-blocking zombie collector for an explicitly-closed child. We own
        these PIDs (pty.fork, not asyncio subprocess), so a manual waitpid is safe.
        Reschedules on the event loop; falls back to the running loop when the
        terminal was closed before it was ever attached (self._loop unset)."""
        try:
            done, _ = os.waitpid(pid, os.WNOHANG)
        except (ChildProcessError, OSError):
            return  # already reaped / never existed
        if done != 0 or attempts <= 0:
            return
        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
        if loop is not None:
            loop.call_later(0.15, self._reap_pid, pid, attempts - 1)

    def shutdown(self):
        for tid in list(self.terms):
            self.close(tid)

    # -- io ---------------------------------------------------------------
    def write(self, tid: str, data: bytes):
        t = self.terms.get(tid)
        if not t:
            return
        try:
            os.write(t.fd, data)
        except OSError:
            pass

    def resize(self, tid: str, rows: int, cols: int):
        t = self.terms.get(tid)
        if not t:
            return
        t.rows, t.cols = rows, cols
        self._set_winsize(t.fd, rows, cols)

    def attach(self, tid: str, queue: "asyncio.Queue") -> bytes:
        """Register a sink queue and return the caught-up scrollback buffer."""
        t = self.terms.get(tid)
        if not t:
            return b""
        self._ensure_reader(t)
        t.clients.add(queue)
        return bytes(t.buffer)

    def detach(self, tid: str, queue: "asyncio.Queue"):
        t = self.terms.get(tid)
        if t:
            t.clients.discard(queue)

    def _ensure_reader(self, t: Terminal):
        if t._reader_installed:
            return
        self._loop = asyncio.get_event_loop()
        self._loop.add_reader(t.fd, self._on_readable, t)
        t._reader_installed = True

    def _reap(self, t: Terminal):
        """Full cleanup of a dead PTY: drop the reader, signal clients, close the
        fd, reap the child, and forget the terminal. Idempotent."""
        if self._loop:
            try:
                self._loop.remove_reader(t.fd)
            except Exception:
                pass
        for q in list(t.clients):
            try:
                q.put_nowait(None)
            except Exception:
                pass
        try:
            os.close(t.fd)
        except OSError:
            pass
        try:
            os.waitpid(t.pid, os.WNOHANG)  # reap the zombie if it has exited
        except (ChildProcessError, OSError):
            pass
        self.terms.pop(t.id, None)

    def _on_readable(self, t: Terminal):
        try:
            data = os.read(t.fd, 65536)
        except OSError:
            data = b""
        if not data:  # EOF — shell exited on its own (e.g. `exit`)
            self._reap(t)
            return
        t.buffer.extend(data)
        if len(t.buffer) > SCROLLBACK_BYTES:
            del t.buffer[: len(t.buffer) - SCROLLBACK_BYTES]
        for q in list(t.clients):
            try:
                q.put_nowait(data)
            except Exception:
                pass
