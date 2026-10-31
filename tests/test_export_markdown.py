#!/usr/bin/env python3
"""Tests for the Markdown export (export_markdown.py, /api/export-markdown, and
its command line), plus the parent→sub-agent links the outline graph uses."""

from __future__ import annotations

import io
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from urllib.parse import quote

import chatgpt_parser as chatgpt
import claude_parser as claude
import export_markdown
from tests.fixture_builders import (
    ViewerServerTestCase,
    _assistant,
    _user,
    _write_chatgpt_export,
    _write_jsonl,
    chatgpt_conversation,
)


def _session(events, **extra):
    data = {"agent": "claude", "id": "abc12345-0000", "title": "Demo", "meta": {"model": "claude-test"},
            "events": events}
    data.update(extra)
    return data


def _headings(md: str) -> list[str]:
    """Headings outside fenced code blocks."""
    out, fence = [], None
    for line in md.splitlines():
        m = re.match(r"^(`{3,})", line)
        if m:
            if fence is None:
                fence = m.group(1)
            elif line.strip() == fence:
                fence = None
            continue
        if fence is None and line.startswith("#"):
            out.append(line)
    return out


class MarkdownRenderingTests(unittest.TestCase):
    def test_fence_outgrows_backticks_in_content(self):
        fenced = export_markdown.fence("a ```` b")
        self.assertTrue(fenced.startswith("`````\n") and fenced.endswith("\n`````"))

    def test_front_matter_and_speaker_sections(self):
        md = export_markdown.session_markdown(_session([
            {"kind": "user", "ts": "2026-01-01T00:00:00Z", "blocks": [{"type": "text", "text": "hello"}]},
            {"kind": "assistant", "ts": "2026-01-01T00:00:01Z", "blocks": [{"type": "text", "text": "hi"}]},
            {"kind": "assistant", "ts": "2026-01-01T00:00:02Z", "blocks": [{"type": "text", "text": "more"}]},
            {"kind": "user", "ts": "2026-01-01T00:00:03Z", "blocks": [{"type": "text", "text": "again"}]},
        ]), exported_at="2026-01-02T00:00:00+00:00")
        self.assertTrue(md.startswith('---\ntitle: "Demo"\nagent: "claude"\nsession_id: "abc12345-0000"\n'))
        self.assertIn('started: "2026-01-01T00:00:00Z"', md)
        # Consecutive assistant records share one section.
        self.assertEqual(_headings(md), ["# Demo", "## User · #1", "## Assistant", "## User · #2"])
        self.assertIn("*2026-01-01 00:00:01 UTC · claude-test*", md)

    def test_tool_call_thinking_and_reply_subsections(self):
        long_output = "x" * 50
        md = export_markdown.session_markdown(_session([
            {"kind": "assistant", "blocks": [
                {"type": "thinking", "text": "line one\nline two"},
                {"type": "tool_use", "name": "Bash", "input": {"command": "ls ```"},
                 "result": {"text": long_output, "is_error": True}},
                {"type": "text", "text": "done"},
            ]},
        ]), max_result_chars=10)
        self.assertEqual(_headings(md), ["# Demo", "## Assistant", "### Thinking", "### Tool: Bash",
                                         "#### Error", "### Reply"])
        self.assertIn("> line one\n> line two", md)
        self.assertIn("````sh\nls ```\n````", md)
        self.assertIn("[… 40 more characters truncated]", md)

    def test_codex_flat_shape(self):
        md = export_markdown.session_markdown(_session([
            {"kind": "user", "text": "go"},
            {"kind": "reasoning", "text": "", "has_encrypted": True},
            {"kind": "tool", "name": "shell", "input": {"command": ["git", "status"]}, "summary": "",
             "result": {"text": "", "output": "clean"}},
            {"kind": "assistant", "text": "all clean"},
            {"kind": "tokens", "usage": {}},
        ], agent="codex"))
        self.assertEqual(_headings(md), ["# Demo", "## User · #1", "## Assistant", "### Reasoning (summary)",
                                         "### Tool: shell", "#### Result", "### Reply"])
        self.assertIn("```sh\ngit status\n```", md)
        self.assertIn("```text\nclean\n```", md)

    def test_branch_renders_deeper_without_taking_a_prompt_number(self):
        md = export_markdown.session_markdown(_session([
            {"kind": "branch", "count": 1, "groups": [[
                {"kind": "user", "blocks": [{"type": "text", "text": "draft"}]},
            ]]},
            {"kind": "user", "blocks": [{"type": "text", "text": "final"}]},
        ]))
        self.assertEqual(_headings(md), ["# Demo", "## Abandoned branch (1 alternative)", "### Alternative 1",
                                         "#### User · #1", "## User · #1"])

    def test_chatgpt_conversation(self):
        session = chatgpt.build_session(chatgpt_conversation())
        md = export_markdown.session_markdown(session)
        self.assertIn('agent: "chatgpt"', md)
        self.assertIn("### Tool: python", md)
        self.assertIn("*[image: screenshot.png]*", md)
        self.assertIn("[the docs](https://netboot.xyz/docs)", md)


class MarkdownEndpointTests(ViewerServerTestCase):
    def test_export_markdown_route(self):
        for file_id in (str(self.fixture), self.chatgpt_file):
            status, headers, body = self.get("/api/export-markdown?file=" + quote(file_id))
            self.assertEqual(status, 200, body[:200])
            self.assertTrue(headers["Content-Type"].startswith("text/markdown"))
            self.assertRegex(headers["Content-Disposition"], r'^attachment; filename="[A-Za-z0-9-]+\.md"$')
            self.assertTrue(body.startswith(b"---\n"))

    def test_export_markdown_needs_a_known_session(self):
        self.assertEqual(self.get("/api/export-markdown")[0], 400)
        self.assertIn(self.get("/api/export-markdown?file=chatgpt:no-such-id")[0], (403, 404))


class MarkdownCliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._old_paths = list(chatgpt.EXPORT_PATHS)

    def tearDown(self):
        chatgpt.configure(self._old_paths)
        self._tmp.cleanup()

    def test_converts_every_conversation_in_an_export(self):
        second = chatgpt_conversation("c0ffee00-0000-4000-8000-000000000002", "Second chat")
        export = _write_chatgpt_export(self.tmp / "export.zip", [chatgpt_conversation(), second])
        out = self.tmp / "md"
        with redirect_stdout(io.StringIO()):
            status = export_markdown.main(["--chatgpt-export", str(export), "--out", str(out)])
        self.assertEqual(status, 0)
        names = sorted(p.name for p in out.iterdir())
        self.assertEqual(names, ["2026-01-01-chatgpt-netboot-planning-c0ffee00.md",
                                 "2026-01-01-chatgpt-second-chat-c0ffee00.md", "INDEX.md"])
        self.assertIn("Second chat", (out / "INDEX.md").read_text(encoding="utf-8"))


class SubagentLinkTests(unittest.TestCase):
    """A Claude Task/Agent call links to the sub-agent transcript it spawned."""

    def test_task_call_gets_child_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "-proj" / "sess.jsonl"
            parent.parent.mkdir()
            call = _assistant("a1", "u1", "", "2026-01-01T00:00:01Z")
            call["message"]["content"] = [{
                "type": "tool_use", "id": "toolu_1", "name": "Agent",
                "input": {"description": "Find files", "subagent_type": "Explore",
                          "prompt": "Look for   the config\nfiles"},
            }]
            _write_jsonl(parent, [_user("u1", None, "start", "2026-01-01T00:00:00Z"), call])
            child = parent.parent / "sess" / "subagents" / "agent-1.jsonl"
            unrelated = parent.parent / "sess" / "subagents" / "agent-2.jsonl"
            child.parent.mkdir(parents=True)
            _write_jsonl(child, [_user("s1", None, "Look for the config files", "2026-01-01T00:00:02Z")])
            _write_jsonl(unrelated, [_user("s2", None, "something else", "2026-01-01T00:00:03Z")])
            data = claude.parse_session(parent)
            blocks = [b for e in data["events"] for b in e.get("blocks") or [] if b.get("type") == "tool_use"]
            self.assertEqual(blocks[0].get("child_file"), str(child))


if __name__ == "__main__":
    unittest.main()
