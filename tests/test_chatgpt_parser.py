#!/usr/bin/env python3
"""Tests for chatgpt_parser: finding exports, walking the conversation tree,
and mapping ChatGPT's message types onto the viewer's block event shape."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

import chatgpt_parser as chatgpt
import event_schema
from tests.fixture_builders import TINY_PNG, _write_chatgpt_export, chatgpt_conversation

CONV_ID = chatgpt_conversation()["id"]


class ChatGPTParserTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._old_paths = list(chatgpt.EXPORT_PATHS)

    def tearDown(self):
        chatgpt.configure(self._old_paths)
        self._tmp.cleanup()

    def load(self, path: Path, **kwargs) -> dict:
        chatgpt.configure(path)
        data = chatgpt.parse_session_by_id(CONV_ID, **kwargs)
        self.assertIsNotNone(data)
        return data


class DiscoveryTests(ChatGPTParserTestCase):
    def test_zip_extracted_dir_and_folder_of_exports(self):
        zipped = _write_chatgpt_export(self.tmp / "a" / "export.zip")
        extracted = _write_chatgpt_export(self.tmp / "b" / "export", extracted=True)
        self.assertEqual([e.path for e in chatgpt.discover_exports([zipped])], [zipped])
        self.assertEqual([e.path for e in chatgpt.discover_exports([extracted])], [extracted])
        # A folder that holds exports (here the zip's parent) is scanned one level down.
        self.assertEqual([e.path for e in chatgpt.discover_exports([zipped.parent])], [zipped])

    def test_unrelated_zip_and_missing_paths_are_ignored(self):
        other = self.tmp / "other.zip"
        with zipfile.ZipFile(other, "w") as zf:
            zf.writestr("readme.txt", "not an export")
        self.assertEqual(chatgpt.discover_exports([other, self.tmp / "missing"]), [])
        chatgpt.configure(self.tmp / "missing")
        self.assertEqual(chatgpt.list_sessions(), [])

    def test_export_nested_in_a_zip_folder(self):
        path = self.tmp / "nested.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("chatgpt-export/conversations.json", "[]")
        export = chatgpt.open_export(path)
        self.assertEqual((export.prefix, export.conversation_files), ("chatgpt-export/", ["conversations.json"]))

    def test_sharded_conversation_files_are_all_read(self):
        second = chatgpt_conversation("c0ffee00-0000-4000-8000-000000000002", "Second")
        chatgpt.configure(_write_chatgpt_export(
            self.tmp / "export.zip", [chatgpt_conversation(), second], sharded=True))
        self.assertEqual({s["title"] for s in chatgpt.list_sessions()}, {"Netboot planning", "Second"})

    def test_newest_copy_wins_across_exports(self):
        old = chatgpt_conversation(title="Old title", update_time=1767225700.0)
        new = chatgpt_conversation(title="New title", update_time=1767229999.0)
        _write_chatgpt_export(self.tmp / "exports" / "1-new.zip", [new])
        _write_chatgpt_export(self.tmp / "exports" / "2-old.zip", [old])
        chatgpt.configure(self.tmp / "exports")
        self.assertEqual([s["title"] for s in chatgpt.list_sessions()], ["New title"])


class SummaryTests(ChatGPTParserTestCase):
    def test_summary_shape_and_counts(self):
        chatgpt.configure(_write_chatgpt_export(self.tmp / "export.zip"))
        (summary,) = chatgpt.list_sessions()
        self.assertEqual(event_schema.validate_summary(summary), [])
        self.assertEqual(summary["file"], "chatgpt:" + CONV_ID)
        self.assertEqual(summary["model"], "gpt-5-thinking")
        # The abandoned first draft is not counted; the two live prompts are.
        # One python call and one web lookup, the lookup also counted as web.
        self.assertEqual((summary["n_user"], summary["n_tool"], summary["n_web"]), (2, 2, 1))
        self.assertTrue(summary["first_ts"].startswith("2026-01-01T00:00:00"))


class ConversationTests(ChatGPTParserTestCase):
    def setUp(self):
        super().setUp()
        self.data = self.load(_write_chatgpt_export(self.tmp / "export.zip"))
        self.events = self.data["events"]

    def test_session_conforms_to_the_event_schema(self):
        self.assertEqual(event_schema.validate_session(self.data), [])

    def test_active_path_with_abandoned_branch_at_the_fork(self):
        kinds = [e["kind"] for e in self.events]
        self.assertEqual(kinds, ["instructions", "branch", "user", "assistant", "user", "assistant"])
        branch = self.events[1]
        self.assertEqual(branch["count"], 1)
        self.assertIn("abandonedword", branch["groups"][0][0]["blocks"][0]["text"])

    def test_custom_instructions_and_hidden_messages(self):
        ctx = self.events[0]
        self.assertEqual(ctx["label"], "Custom instructions")
        self.assertIn("Be terse.", ctx["text"])

    def test_assistant_turn_merges_thoughts_tool_and_reply(self):
        turn = self.events[3]
        self.assertEqual([b["type"] for b in turn["blocks"]], ["thinking", "tool_use", "tool_use", "text", "text"])
        thinking, tool, _search, reply, _sources = turn["blocks"]
        self.assertIn("**Planning**", thinking["text"])
        self.assertEqual((tool["name"], tool["input"], tool["result"]["text"]),
                         ("python", {"code": "print(2 + 2)"}, "4"))
        self.assertEqual(turn["model"], "gpt-5-thinking")
        self.assertIn("[dnsmasq man page](https://example.test/dnsmasq)", reply["text"])
        self.assertIn("[the docs](https://netboot.xyz/docs)", reply["text"])
        self.assertNotRegex(reply["text"], "[\ue000-\uf8ff]")

    def test_citation_resolved_through_item_refs(self):
        # matched_text is empty, as in current exports; the marker key
        # turn0search1 finds the item through its refs, tracking param dropped.
        reply = self.events[3]["blocks"][3]["text"]
        self.assertIn("Pin the image ([iPXE images](https://example.test/ipxe)).", reply)
        self.assertIn("Attributed span.", reply)

    def test_web_search_results_precede_the_reply(self):
        search = self.events[3]["blocks"][2]
        self.assertEqual((search["name"], search["input"]), ("web_search", {"sites": ["example.test", "netboot.xyz"]}))
        result = search["result"]["text"]
        self.assertTrue(result.startswith("3 pages retrieved"))
        self.assertIn("iPXE images\nhttps://example.test/ipxe\nchainload", result)

    def test_sources_list_follows_the_reply(self):
        sources = self.events[3]["blocks"][4]["text"]
        self.assertEqual(sources.splitlines(), [
            "**Sources**",
            "",
            "- [netboot.xyz docs](https://netboot.xyz/docs) · netboot.xyz",
            "- [dnsmasq man page](https://example.test/dnsmasq)",
            "- [iPXE images](https://example.test/ipxe)",
        ])

    def test_uploaded_image_is_inlined(self):
        image = self.events[4]["blocks"][0]
        self.assertEqual(image["type"], "image")
        self.assertTrue(image["data_uri"].startswith("data:image/png;base64,"))
        self.assertNotIn("asset", image)

    def test_text_only_parse_skips_image_files(self):
        data = chatgpt.parse_session_by_id(CONV_ID, with_images=False)
        self.assertEqual(data["events"][4]["blocks"][0], {"type": "text", "text": "*[image: screenshot.png]*"})

    def test_extracted_export_reads_the_same(self):
        extracted = self.load(_write_chatgpt_export(self.tmp / "dir", extracted=True))
        self.assertEqual(extracted["events"][4]["blocks"][0]["data_uri"], self.events[4]["blocks"][0]["data_uri"])


class CleanTextTests(unittest.TestCase):
    def test_marker_kinds(self):
        clean = chatgpt.clean_text
        self.assertEqual(clean("a citeturn0search1b"), "a b")
        self.assertEqual(clean("entity[\"people\",\"Ada Lovelace\",0]"), "Ada Lovelace")
        self.assertEqual(clean("urlLabelturn0search3"), "Label")
        self.assertEqual(clean("x【6†source】y"), "xy")
        self.assertEqual(clean("plain"), "plain")

    def test_missing_asset_becomes_a_note(self):
        conv = chatgpt_conversation()
        conv["mapping"]["u2"]["message"]["content"]["parts"][0]["asset_pointer"] = "sediment://file_missing"
        session = chatgpt.build_session(conv)
        events = chatgpt._hydrate(session["events"], None, {})
        self.assertEqual(events[4]["blocks"][0]["type"], "text")
        self.assertIn("image", events[4]["blocks"][0]["text"])


if __name__ == "__main__":
    unittest.main()
