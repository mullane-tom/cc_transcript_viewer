#!/usr/bin/env python3
"""Render one parsed session as a Markdown file meant for tools and models.

The HTML export (export_html.py) is the viewer frozen around one transcript; this
is the opposite trade — no styling, no scripts, nothing collapsed, just a
predictable structure a script (or another model) can walk:

- YAML front matter with the session's identity (agent, id, model, cwd, times).
- One ``## User · #N`` or ``## Assistant`` section per change of speaker, each
  followed by an italic ``*timestamp · model*`` line. Consecutive assistant
  records (a tool loop) stay under one heading.
- Inside an assistant section: reply text as-is, ``### Thinking`` as a
  blockquote, and ``### Tool: <name>`` with the input and its ``#### Result``
  (or ``#### Error``) in fenced blocks. Fences grow past any backtick run in
  their content, so tool output can't break out of them.
- Abandoned branches become ``## Abandoned branch`` sections whose contents
  are rendered two heading levels deeper.

Run as a script to convert sessions in bulk, e.g. a whole ChatGPT export::

    python3 export_markdown.py --chatgpt-export ~/Downloads/chatgpt-export.zip --out ~/chatgpt-md
    python3 export_markdown.py ~/.claude/projects/-Users-me-proj/SESSION.jsonl --out .
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import export_html

# Tool results longer than this are cut (with a note saying how much). A
# transcript is mostly tool output by volume; whole file dumps and build logs
# drown the conversation a reader of the Markdown is usually after.
DEFAULT_MAX_RESULT_CHARS = 10_000

# Codex/turn bookkeeping that the HTML view folds away too.
_SKIPPED_KINDS = {"status", "context", "tokens", "raw"}


def fence(text: str, lang: str = "") -> str:
    """`text` in a backtick fence longer than any backtick run inside it."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{lang}\n{text.rstrip(chr(10))}\n{ticks}"


def quote(text: str) -> str:
    return "\n".join("> " + line if line else ">" for line in text.strip().splitlines())


def _yaml_value(value) -> str:
    # JSON strings are valid YAML double-quoted scalars.
    return json.dumps(value, ensure_ascii=False)


def _ts(ts) -> str:
    if not isinstance(ts, str) or not ts:
        return ""
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return ts
    if parsed.tzinfo:
        parsed = parsed.astimezone(timezone.utc)
        return parsed.strftime("%Y-%m-%d %H:%M:%S UTC")
    return parsed.strftime("%Y-%m-%d %H:%M:%S")


def _truncate(text: str, limit: int | None) -> str:
    if limit is None or limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n[… {len(text) - limit:,} more characters truncated]"


def _tool_input(name: str, value) -> str:
    """The most readable rendering of a tool call's input."""
    if isinstance(value, str):
        lang = "diff" if name in ("apply_patch",) or value.startswith("*** Begin Patch") else ""
        return fence(value, lang)
    if not isinstance(value, dict):
        return fence(json.dumps(value, indent=2, ensure_ascii=False), "json")
    command = value.get("command") or value.get("cmd")
    if isinstance(command, list) and all(isinstance(c, str) for c in command):
        command = shlex.join(command)
    out = []
    if isinstance(command, str) and command:
        if value.get("description"):
            out.append(f"*{value['description']}*")
        out.append(fence(command, "sh"))
        return "\n\n".join(out)
    if "old_string" in value or "new_string" in value:
        path = value.get("file_path") or value.get("path") or ""
        diff = [f"--- {path}", f"+++ {path}"]
        diff += ["-" + line for line in str(value.get("old_string") or "").splitlines()]
        diff += ["+" + line for line in str(value.get("new_string") or "").splitlines()]
        return fence("\n".join(diff), "diff")
    if isinstance(value.get("prompt"), str) and (value.get("description") or value.get("subagent_type")):
        sub = value.get("subagent_type") or value.get("agentType") or ""
        head = (f"[{sub}] " if sub else "") + (value.get("description") or "")
        return f"*{head.strip()}*\n\n" + fence(value["prompt"], "text")
    if isinstance(value.get("code"), str) and len(value) == 1:
        return fence(value["code"])
    if isinstance(value.get("text"), str) and len(value) == 1:
        return fence(value["text"], "text")
    return fence(json.dumps(value, indent=2, ensure_ascii=False), "json")


class _Writer:
    def __init__(self, max_result_chars: int | None, depth: int = 0, counter: list | None = None):
        self.out: list[str] = []
        self.max_result_chars = max_result_chars
        self.depth = depth  # extra heading levels (2 per nested branch)
        self.role: str | None = None
        # True after a ### subsection (thinking, tool) inside the open speaker
        # section: reply text that follows gets its own ### Reply heading so it
        # isn't read as part of that subsection.
        self.in_sub = False
        self.user_counter = counter if counter is not None else [0]

    def h(self, level: int, text: str) -> None:
        self.out.append("#" * min(6, level + self.depth) + " " + text)
        self.in_sub = level >= 3

    def para(self, text: str) -> None:
        if text and text.strip():
            if self.in_sub:
                self.h(3, "Reply")
                self.in_sub = False
            self.out.append(text.strip("\n"))

    def section(self, role: str, title: str, ev: dict, model: str = "") -> None:
        """Start a `## title` section unless this speaker's section is open."""
        if self.role == role and role == "assistant":
            return
        self.role = role
        self.h(2, title)
        stamp = " · ".join(x for x in (_ts(ev.get("ts")), model) if x)
        if ev.get("is_sidechain"):
            stamp = (stamp + " · " if stamp else "") + "sub-agent"
        if stamp:
            self.out.append(f"*{stamp}*")

    # -- blocks ---------------------------------------------------------------
    def thinking(self, text: str, summary_only: bool = False, title: str = "Thinking") -> None:
        if not (text or "").strip() and not summary_only:
            return
        self.h(3, title + (" (summary)" if summary_only else ""))
        if (text or "").strip():
            self.out.append(quote(text))
        else:
            self.out.append("*(encrypted; not readable from the transcript)*")

    def tool(self, name: str, value, result, *, child_file: str = "", error: bool = False) -> None:
        self.h(3, f"Tool: {name or 'tool'}")
        if value not in (None, {}, ""):
            self.out.append(_tool_input(name, value))
        if child_file:
            self.out.append(f"Sub-agent transcript: `{child_file}`")
        if result is None:
            return
        if isinstance(result, dict):
            # Block-shaped results carry `text`; Codex's flat ones `output`.
            text = result.get("text") or result.get("output") or ""
            error = error or bool(result.get("is_error"))
            n_images = len(result.get("images") or [])
        else:
            text, n_images = str(result), 0
        if isinstance(text, (dict, list)):
            text = json.dumps(text, indent=2, ensure_ascii=False)
        self.h(4, "Error" if error else "Result")
        if text:
            self.out.append(fence(_truncate(str(text), self.max_result_chars), "text"))
        if n_images:
            self.out.append(f"*[{n_images} image{'s' if n_images != 1 else ''}]*")
        if not text and not n_images:
            self.out.append("*(no output)*")

    def blocks(self, blocks: list) -> None:
        for b in blocks:
            btype = b.get("type")
            if btype == "text":
                self.para(b.get("text") or "")
            elif btype == "thinking":
                self.thinking(b.get("text") or "", bool(b.get("has_encrypted")))
            elif btype == "image":
                label = b.get("name") or b.get("media_type") or "image"
                self.out.append(f"*[image: {label}]*")
            elif btype == "tool_use":
                self.tool(b.get("name"), b.get("input"), b.get("result"), child_file=b.get("child_file") or "")

    # -- events ---------------------------------------------------------------
    def event(self, ev: dict, session_model: str) -> None:
        kind = ev.get("kind")
        if kind in _SKIPPED_KINDS:
            return
        if kind == "user":
            self.user_counter[0] += 1
            self.section("user", f"User · #{self.user_counter[0]}", ev)
            if ev.get("blocks"):
                self.blocks(ev["blocks"])
            else:
                self.para(ev.get("text") or "")
                n_images = len(ev.get("images") or []) + len(ev.get("local_images") or [])
                if n_images:
                    self.out.append(f"*[{n_images} image{'s' if n_images != 1 else ''} attached]*")
        elif kind == "assistant":
            self.section("assistant", "Assistant", ev, ev.get("model") or session_model)
            if ev.get("blocks"):
                self.blocks(ev["blocks"])
            else:
                self.para(ev.get("text") or "")
        elif kind == "reasoning":
            self.section("assistant", "Assistant", ev, session_model)
            self.thinking(ev.get("text") or "", bool(ev.get("has_encrypted")), "Reasoning")
        elif kind == "tool":
            self.section("assistant", "Assistant", ev, session_model)
            self.tool(ev.get("name"), ev.get("input"), ev.get("result"))
        elif kind in ("web_search", "web_call"):
            self.section("assistant", "Assistant", ev, session_model)
            self.h(3, "Web search")
            action = ev.get("action") if isinstance(ev.get("action"), dict) else {}
            queries = [ev.get("query")] + list(action.get("queries") or []) + [action.get("query"), action.get("url")]
            seen = []
            for q in queries:
                if isinstance(q, str) and q and q not in seen:
                    seen.append(q)
            self.out.append("\n".join(f"- {q}" for q in seen) or "*(no query recorded)*")
        elif kind == "instructions":
            self.role = None
            self.h(2, ev.get("label") or "Instructions")
            self.out.append(fence(ev.get("text") or "", "text"))
        elif kind == "system":
            self.role = None
            if ev.get("subtype") == "compact_boundary":
                stats = ev.get("compaction") or {}
                detail = ", ".join(
                    f"{k}: {v}" for k, v in stats.items() if isinstance(v, (str, int, float)) and v not in ("", None)
                )
                self.out.append("---")
                self.out.append(f"*Conversation compacted{' (' + detail + ')' if detail else ''}*")
            elif (ev.get("text") or "").strip():
                self.out.append(quote("**System:** " + ev["text"]))
        elif kind == "notice":
            self.out.append(quote(f"**{ev.get('label') or 'Notice'}:** " + (ev.get("text") or "")))
        elif kind == "attachment":
            self.attachment(ev)
        elif kind == "guardian_request":
            self.role = None
            self.h(2, "Guardian review")
            self.out.append(fence(json.dumps(ev.get("request") or {}, indent=2, ensure_ascii=False), "json"))
        elif kind == "guardian_decision":
            self.role = None
            reason = ev.get("rationale") or ev.get("reason") or ""
            self.out.append(f"**Guardian decision:** {ev.get('outcome')}" + (f" — {reason}" if reason else ""))
        elif kind == "branch":
            self.branch(ev, session_model)

    def attachment(self, ev: dict) -> None:
        att = ev.get("att_type") or "attachment"
        what = ev.get("display_path") or ev.get("filename") or ev.get("hook_name") or ev.get("command") or ""
        line = f"📎 **{att}**" + (f": `{what}`" if what else "")
        self.out.append(quote(line))
        body = ev.get("content") or ev.get("stdout") or ""
        if isinstance(body, str) and body.strip():
            self.out.append(fence(_truncate(body, self.max_result_chars), "text"))

    def branch(self, ev: dict, session_model: str) -> None:
        self.role = None
        groups = ev.get("groups") or []
        self.h(2, f"Abandoned branch ({len(groups)} alternative{'s' if len(groups) != 1 else ''})")
        for i, group in enumerate(groups, 1):
            self.h(3, f"Alternative {i}")
            # User prompts in an abandoned branch don't take a number in the
            # main sequence (the outline doesn't count them either).
            sub = _Writer(self.max_result_chars, self.depth + 2, counter=[0])
            for sub_ev in group:
                sub.event(sub_ev, session_model)
            self.out.extend(sub.out)
        self.role = None


def _front_matter(data: dict, exported_at: str) -> list[str]:
    meta = data.get("meta") or {}
    fields = {
        "title": data.get("title") or "",
        "agent": data.get("agent") or "",
        "session_id": data.get("id") or "",
        "source": data.get("file") or data.get("source_file") or "",
        "model": meta.get("model") or "",
        "cwd": meta.get("cwd") or "",
        "git_branch": meta.get("git_branch") or "",
        "started": _first_ts(data.get("events") or [], first=True),
        "last_activity": _first_ts(data.get("events") or [], first=False),
        "parent_session": data.get("parent_file") if data.get("is_subagent") else "",
        "subagent_type": data.get("subagent_type") or "",
        "exported_at": exported_at,
    }
    lines = ["---"]
    for key, value in fields.items():
        if value:
            lines.append(f"{key}: {_yaml_value(value)}")
    lines.append("---")
    return lines


def _first_ts(events: list, first: bool) -> str:
    stamps = [ev.get("ts") for ev in events if isinstance(ev.get("ts"), str) and ev.get("ts")]
    if not stamps:
        return ""
    return stamps[0] if first else stamps[-1]


def session_markdown(
    data: dict,
    *,
    exported_at: str | None = None,
    max_result_chars: int | None = DEFAULT_MAX_RESULT_CHARS,
) -> str:
    """One parsed session (any agent) as a Markdown document."""
    exported_at = exported_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    writer = _Writer(max_result_chars)
    model = (data.get("meta") or {}).get("model") or ""
    for ev in data.get("events") or []:
        writer.event(ev, model)
    title = (data.get("title") or "(untitled session)").strip().replace("\n", " ")
    front = "\n".join(_front_matter(data, exported_at))
    return "\n\n".join([front, "# " + title] + writer.out) + "\n"


def export_filename(data: dict, suffix: str = ".md") -> str:
    """`<agent>-<slug>-<id8>.md`, the same naming the HTML export uses."""
    return export_html.export_filename(data)[: -len(".html")] + suffix


def dated_filename(data: dict, first_ts: str = "") -> str:
    """`YYYY-MM-DD-<agent>-<slug>-<id8>.md`, for bulk exports that sort by date."""
    stamp = (first_ts or _first_ts(data.get("events") or [], first=True))[:10]
    name = export_filename(data)
    return f"{stamp}-{name}" if re.match(r"^\d{4}-\d{2}-\d{2}$", stamp) else name


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def _write_index(out_dir: Path, rows: list[tuple[str, str, str, str]]) -> Path:
    lines = ["# Exported transcripts", "", "| Date | Agent | Title | File |", "|---|---|---|---|"]
    for date, agent, title, name in sorted(rows, reverse=True):
        safe_title = title.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {date} | {agent} | {safe_title} | [{name}]({name}) |")
    index = out_dir / "INDEX.md"
    index.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export transcripts as Markdown files.")
    ap.add_argument("sessions", nargs="*",
                    help="transcript paths or session ids (e.g. chatgpt:<id>, opencode:<id>)")
    ap.add_argument("--chatgpt-export", type=Path, action="append", default=[],
                    help="ChatGPT export .zip or directory; every conversation in it is exported")
    ap.add_argument("--out", type=Path, default=Path("."), help="output directory (default: .)")
    ap.add_argument("--max-result-chars", type=int, default=DEFAULT_MAX_RESULT_CHARS,
                    help="truncate each tool result to this many characters (0 = no limit)")
    args = ap.parse_args(argv)
    if not args.sessions and not args.chatgpt_export:
        ap.error("give at least one session or --chatgpt-export")

    import chatgpt_parser
    import server  # resolves paths against the default transcript roots

    if args.chatgpt_export:
        chatgpt_parser.configure(args.chatgpt_export)
        if not chatgpt_parser.discover_exports():
            print("no ChatGPT export found at: " + ", ".join(map(str, args.chatgpt_export)), file=sys.stderr)
            return 1

    targets = list(args.sessions)
    for summary in sorted(chatgpt_parser.list_sessions() if args.chatgpt_export else [],
                          key=lambda s: s.get("first_ts") or ""):
        targets.append(summary["file"])

    args.out.mkdir(parents=True, exist_ok=True)
    limit = args.max_result_chars or None
    rows, failed = [], 0
    for target in targets:
        if target.startswith(chatgpt_parser.SESSION_SCHEME):
            data = chatgpt_parser.parse_session_by_id(target[len(chatgpt_parser.SESSION_SCHEME):], with_images=False)
        else:
            data = server.load_session(target)
        if data is None:
            print(f"skipped (not found or outside the transcript roots): {target}", file=sys.stderr)
            failed += 1
            continue
        data.setdefault("file", target)
        name = dated_filename(data)
        (args.out / name).write_text(session_markdown(data, max_result_chars=limit), encoding="utf-8")
        date = _first_ts(data.get("events") or [], first=True)[:10]
        rows.append((date, data.get("agent") or "", data.get("title") or "", name))
    if len(rows) > 1:
        _write_index(args.out, rows)
    print(f"wrote {len(rows)} file{'s' if len(rows) != 1 else ''} to {args.out}"
          + (f" ({failed} skipped)" if failed else ""))
    return 1 if failed and not rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
