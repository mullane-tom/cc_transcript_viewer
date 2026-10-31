#!/usr/bin/env python3
"""Tests for web_lookup: every agent's web searches and fetches mapped onto
one `web` shape, and that shape's Markdown rendering."""

from __future__ import annotations

import json
import unittest

import chatgpt_parser as chatgpt
import export_markdown
import web_lookup
from tests.fixture_builders import chatgpt_conversation

CLAUDE_SEARCH_RESULT = (
    'Web search results for query: "netboot secure boot"\n\n'
    'Links: [{"title":"netboot.xyz 3.0","url":"https://netboot.xyz/blog/3.0"},'
    '{"title":"Secure Boot - iPXE","url":"https://ipxe.org/appnote/secureboot"}]\n\n'
    "netboot.xyz 3.x ships a signed shim."
)


class FromToolTests(unittest.TestCase):
    def test_claude_web_search(self):
        web = web_lookup.from_tool("WebSearch", {"query": "netboot secure boot"}, {"text": CLAUDE_SEARCH_RESULT})
        self.assertEqual(web["kind"], "search")
        self.assertEqual(web["queries"], ["netboot secure boot"])
        self.assertEqual([p["url"] for p in web["pages"]],
                         ["https://netboot.xyz/blog/3.0", "https://ipxe.org/appnote/secureboot"])
        self.assertEqual((web["text"], web["text_label"]), ("netboot.xyz 3.x ships a signed shim.", "Summary"))

    def test_claude_web_fetch(self):
        web = web_lookup.from_tool("WebFetch", {"url": "https://ipxe.org/", "prompt": "Secure Boot?"},
                                   {"text": "Yes, via shim."})
        self.assertEqual((web["kind"], web["url"], web["prompt"]), ("fetch", "https://ipxe.org/", "Secure Boot?"))
        self.assertEqual((web["text"], web["text_label"]), ("Yes, via shim.", "Answer"))
        self.assertEqual(web["pages"], [{"title": "https://ipxe.org/", "url": "https://ipxe.org/"}])

    def test_cursor_web_search_json_result(self):
        result = json.dumps({"references": [{"title": "T", "url": "https://e.test/", "chunk": "c"}]})
        web = web_lookup.from_tool("WebSearch", {"search_term": "q"}, {"text": result})
        self.assertEqual(web["queries"], ["q"])
        self.assertEqual(web["pages"], [{"title": "T", "url": "https://e.test/", "snippet": "c"}])
        self.assertNotIn("text", web)

    def test_opencode_websearch_and_webfetch(self):
        out = "Title: A\nURL: https://a.test/\nText: first\nline two\n\nTitle: B\nURL: https://b.test/\nText: second"
        web = web_lookup.from_tool("websearch", {"query": "q"}, {"text": out})
        self.assertEqual(web["pages"], [
            {"title": "A", "url": "https://a.test/", "snippet": "first line two"},
            {"title": "B", "url": "https://b.test/", "snippet": "second"},
        ])
        self.assertNotIn("text", web)  # nothing left to repeat raw
        fetch = web_lookup.from_tool("webfetch", {"url": "https://a.test/", "format": "markdown"}, {"text": "# Page"})
        self.assertEqual((fetch["kind"], fetch["text"], fetch["text_label"]), ("fetch", "# Page", "Content"))

    def test_unrecorded_error_and_unrelated_tools(self):
        pending = web_lookup.from_tool("WebSearch", {"query": "q"}, None)
        self.assertEqual((pending["recorded"], pending["pages"]), (False, []))
        failed = web_lookup.from_tool("WebFetch", {"url": "https://x.test/"}, {"text": "403", "is_error": True})
        self.assertTrue(failed["error"])
        self.assertIsNone(web_lookup.from_tool("Bash", {"command": "ls"}, {"text": ""}))

    def test_only_http_urls_become_pages(self):
        web = web_lookup.from_tool("WebFetch", {"url": "javascript:alert(1)"}, {"text": "x"})
        self.assertEqual(web["pages"], [])


class CodexTests(unittest.TestCase):
    def test_search_queries_without_results(self):
        web = web_lookup.from_codex({"kind": "web_call", "query": "", "action": {"type": "search", "queries": ["a", "b"]}})
        self.assertEqual((web["kind"], web["queries"], web["recorded"]), ("search", ["a", "b"], False))
        self.assertIn("not the results", web["note"])

    def test_open_page_and_find_in_page_are_fetches(self):
        web = web_lookup.from_codex({"kind": "web_call", "action": {
            "type": "find_in_page", "url": "https://x.test/", "pattern": "secure"}})
        self.assertEqual((web["kind"], web["url"], web["pattern"]), ("fetch", "https://x.test/", "secure"))


class AnnotateTests(unittest.TestCase):
    def test_walks_blocks_flat_events_and_branches(self):
        data = {"events": [
            {"kind": "assistant", "blocks": [
                {"type": "tool_use", "name": "WebSearch", "input": {"query": "q"}, "result": {"text": CLAUDE_SEARCH_RESULT}},
                {"type": "tool_use", "name": "Bash", "input": {}, "result": None},
            ]},
            {"kind": "web_search", "query": "codex q", "action": {"type": "search"}},
            {"kind": "branch", "count": 1, "groups": [[
                {"kind": "assistant", "blocks": [
                    {"type": "tool_use", "name": "webfetch", "input": {"url": "https://a.test/"}, "result": None}]},
            ]]},
        ]}
        web_lookup.annotate(data)
        blocks = data["events"][0]["blocks"]
        self.assertEqual(blocks[0]["web"]["kind"], "search")
        self.assertNotIn("web", blocks[1])
        self.assertEqual(data["events"][1]["web"]["queries"], ["codex q"])
        self.assertEqual(data["events"][2]["groups"][0][0]["blocks"][0]["web"]["kind"], "fetch")

    def test_chatgpt_search_keeps_its_own_record(self):
        session = web_lookup.annotate(chatgpt.build_session(chatgpt_conversation()))
        search = session["events"][3]["blocks"][2]
        self.assertEqual(search["web"]["sites"], ["example.test", "netboot.xyz"])
        self.assertEqual(len(search["web"]["pages"]), 3)


class MarkdownTests(unittest.TestCase):
    def test_same_layout_for_every_agent(self):
        data = web_lookup.annotate({"agent": "claude", "id": "x", "title": "T", "meta": {}, "events": [
            {"kind": "assistant", "blocks": [
                {"type": "tool_use", "name": "WebSearch", "input": {"query": "q"}, "result": {"text": CLAUDE_SEARCH_RESULT}},
                {"type": "tool_use", "name": "WebFetch", "input": {"url": "https://ipxe.org/", "prompt": "SB?"},
                 "result": {"text": "Yes."}},
            ]},
            {"kind": "web_call", "action": {"type": "open_page", "url": "https://netboot.xyz/docs"}},
        ]})
        md = export_markdown.session_markdown(data)
        self.assertIn("### Web search: q\n\n#### Results (2)\n\n- [netboot.xyz 3.0](https://netboot.xyz/blog/3.0)", md)
        self.assertIn("#### Summary\n\n```text\nnetboot.xyz 3.x ships a signed shim.\n```", md)
        self.assertIn("### Web fetch: https://ipxe.org/\n\n> **Asked:** SB?\n\n#### Answer", md)
        self.assertIn("### Web fetch: https://netboot.xyz/docs\n\n*Codex records the request", md)


if __name__ == "__main__":
    unittest.main()
