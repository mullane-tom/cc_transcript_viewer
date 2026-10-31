#!/usr/bin/env python3
"""The sidebar's 🌐 count (`n_web`): every source counts web searches and
fetches by the same rule that renders them as web blocks, once each."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import claude_parser as claude
import codex_parser as codex
import cursor_parser as cursor
import web_lookup
from tests.fixture_builders import _assistant, _user, _write_jsonl


def _tool_call(uuid, parent, names, ts):
    rec = _assistant(uuid, parent, "", ts)
    rec["message"]["content"] = [
        {"type": "tool_use", "id": f"{uuid}-{i}", "name": name, "input": {}} for i, name in enumerate(names)
    ]
    return rec


class WebCountTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_rule_matches_every_agents_tool_names(self):
        for name in ("WebSearch", "WebFetch", "web_search", "websearch", "webfetch"):
            self.assertTrue(web_lookup.is_web_tool(name), name)
        for name in ("Bash", "Read", "Task", "", None):
            self.assertFalse(web_lookup.is_web_tool(name), name)

    def test_claude_counts_searches_and_fetches(self):
        path = self.dir / "-proj" / "s.jsonl"
        path.parent.mkdir()
        _write_jsonl(path, [
            _user("u1", None, "look it up", "2026-01-01T00:00:00Z"),
            _tool_call("a1", "u1", ["WebSearch", "Bash"], "2026-01-01T00:00:01Z"),
            _tool_call("a2", "a1", ["WebFetch"], "2026-01-01T00:00:02Z"),
        ])
        summary = claude.session_summary(path)
        self.assertEqual((summary["n_tool"], summary["n_web"]), (3, 2))

    def test_cursor_cli_jsonl_counts_web_tools(self):
        path = self.dir / "proj" / "agent-transcripts" / "abc" / "abc.jsonl"
        path.parent.mkdir(parents=True)
        _write_jsonl(path, [
            {"role": "user", "message": {"content": [{"type": "text", "text": "<user_query>search</user_query>"}]}},
            {"role": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "WebSearch", "input": {"search_term": "q"}},
                {"type": "tool_use", "name": "Shell", "input": {"command": "ls"}},
            ]}},
        ])
        summary = cursor.cli_session_summary(path)
        self.assertEqual((summary["n_tool"], summary["n_web"]), (2, 1))

    def _codex_rollout(self):
        path = self.dir / "rollout.jsonl"
        ts = "2026-01-01T00:00:0{}Z"
        _write_jsonl(path, [
            {"timestamp": ts.format(0), "type": "session_meta", "payload": {"id": "t1", "cwd": "/tmp"}},
            {"timestamp": ts.format(1), "type": "event_msg", "payload": {"type": "user_message", "message": "search"}},
            # One search logged both ways (shared call_id) …
            {"timestamp": ts.format(2), "type": "response_item", "payload": {
                "type": "web_search_call", "call_id": "ws1", "status": "completed",
                "action": {"type": "search", "query": "ipxe"}}},
            {"timestamp": ts.format(3), "type": "event_msg", "payload": {
                "type": "web_search_end", "call_id": "ws1", "query": "ipxe secure boot"}},
            # … and one with only the end event.
            {"timestamp": ts.format(4), "type": "event_msg", "payload": {
                "type": "web_search_end", "call_id": "ws2", "query": "netboot.xyz"}},
        ])
        return path

    def test_codex_counts_each_search_once(self):
        self.assertEqual(codex.session_summary(self._codex_rollout())["n_web"], 2)

    def test_codex_renders_each_search_once(self):
        data = codex.parse_session(self._codex_rollout())
        web = [e for e in data["events"] if e["kind"] in ("web_search", "web_call")]
        self.assertEqual([(e["kind"], e.get("query")) for e in web],
                         [("web_call", "ipxe secure boot"), ("web_search", "netboot.xyz")])


if __name__ == "__main__":
    unittest.main()
