#!/usr/bin/env python3
"""Fixture builders and harness helpers shared by the test suite.

Builders that write minimal-but-valid on-disk transcripts for each source
(Claude Code JSONL, Codex rollout JSONL, Cursor CLI JSONL and store.db, ChatGPT
export zips), plus
the loopback HTTP harness the server-backed test classes share.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import chatgpt_parser as chatgpt
import claude_parser as claude
import codex_parser as codex
import cursor_parser as cursor
import opencode_parser as opencode
import server


def patch_server_files(server_module, tmp: Path):
    """Point the viewer-owned files (custom names, summary cache) into tmp so
    tests never touch the user's real ones. Return all replaced state so the
    caller can restore it even when test modules run in a different order."""
    old_state = (
        server_module.CUSTOM_NAMES_FILE,
        server_module._CUSTOM_NAMES_CACHE,
        server_module.CACHE_FILE,
    )
    server_module.CUSTOM_NAMES_FILE = tmp / "viewer" / "names.json"
    server_module._CUSTOM_NAMES_CACHE = None
    server_module.CACHE_FILE = tmp / "cache" / "summaries.json"
    return old_state


def restore_server_files(server_module, old_state) -> None:
    """Undo ``patch_server_files`` without retaining a deleted temp path."""
    (
        server_module.CUSTOM_NAMES_FILE,
        server_module._CUSTOM_NAMES_CACHE,
        server_module.CACHE_FILE,
    ) = old_state


def start_http_server(handler):
    """Serve `handler` on an ephemeral loopback port in a daemon thread."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, httpd.server_address[1], thread


def stop_http_server(httpd, thread):
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def http_get(port: int, path: str, timeout: float = 10):
    """(status, headers, body) for a GET against the test server; HTTP errors
    come back as a normal tuple instead of raising."""
    url = f"http://127.0.0.1:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        try:
            return e.code, e.headers, e.read()
        finally:
            e.close()


def _send_json(port: int, method: str, path: str, payload: dict, timeout: float = 5):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        try:
            return e.code, e.headers, e.read()
        finally:
            e.close()


class ViewerServerTestCase(unittest.TestCase):
    """A live viewer on a loopback port, with every parser pointed at hermetic
    temp fixtures: four Claude sessions, Codex guardian sessions, Cursor CLI
    JSONL and store.db sessions, and an opencode database."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.projects_dir = tmp / "projects"
        cls.projects_dir.mkdir()
        cls.fixture = _write_fixture_session(cls.projects_dir)
        cls.priority_fixture = _write_fixture_session(
            cls.projects_dir,
            "22222222-2222-2222-2222-222222222222",
            "priorityword in the first user message",
        )
        cls.later_prompt_fixture = _write_fixture_session(
            cls.projects_dir,
            "33333333-3333-3333-3333-333333333333",
            "unrelated opening prompt",
            ("laterpromptword in a subsequent user message",),
        )
        cls.metadata_fixture = _write_fixture_session(
            cls.projects_dir,
            "44444444-4444-4444-4444-444444444444",
            "ordinary opening prompt",
            extra_records=(
                {"type": "ai-title", "aiTitle": "Generated title"},
                {"type": "custom-title", "customTitle": "nativepriority title"},
                {"type": "agent-name", "agentName": "reviewer"},
                {"type": "pr-link", "prNumber": 42,
                 "prUrl": "https://example.test/org/repo/pull/42",
                 "prRepository": "org/repo"},
                {"type": "system", "subtype": "compact_boundary",
                 "timestamp": "2024-01-01T00:01:00Z", "content": "Conversation compacted",
                 "compactMetadata": {
                     "trigger": "manual", "preTokens": 12000, "postTokens": 3500,
                     "durationMs": 1250, "preservedMessages": {"uuids": ["a", "b"]},
                     "preCompactDiscoveredTools": ["Read", "Edit"],
                 }},
            ),
        )
        cls._old_parser_config = (
            claude.PROJECTS_DIR,
            codex.CODEX_HOME,
            (cursor.DB_PATH, cursor.PROJECTS_DIR, cursor.CHATS_DIR),
            opencode.DB_PATH,
            list(chatgpt.EXPORT_PATHS),
        )
        claude.configure(cls.projects_dir)
        # Hermetic viewer-owned files — never touch the user's real names.json
        # or ~/.cache summary file from the test suite.
        cls._old_server_files = patch_server_files(server, tmp)
        cls.codex_parent, cls.codex_guardian, cls.codex_image = _write_guardian_sessions(tmp / "codex")
        codex.configure(tmp / "codex")
        cls.cursor_projects = tmp / "cursor-projects"
        cls.cli_fixture = _write_cli_session(cls.cursor_projects)
        cls.cursor_chats = tmp / "cursor-chats"
        cls.cli_store_fixture = _write_cli_store(cls.cursor_chats)
        cursor.configure(
            tmp / "cursor",
            projects_dir=cls.cursor_projects,
            chats_dir=cls.cursor_chats,
        )
        cls.opencode_db = tmp / "opencode" / "opencode.db"
        cls.opencode_parent, cls.opencode_child = _write_opencode_db(cls.opencode_db)
        opencode.configure(cls.opencode_db)
        cls.chatgpt_export = _write_chatgpt_export(tmp / "chatgpt" / "export.zip")
        cls.chatgpt_file = chatgpt.SESSION_SCHEME + chatgpt_conversation()["id"]
        chatgpt.configure(cls.chatgpt_export)

        # Serve on an ephemeral loopback port in a background thread.
        cls.httpd, cls.port, cls.thread = start_http_server(server.Handler)

    @classmethod
    def tearDownClass(cls):
        stop_http_server(cls.httpd, cls.thread)
        restore_server_files(server, cls._old_server_files)
        claude.configure(cls._old_parser_config[0])
        codex.configure(cls._old_parser_config[1])
        cursor.configure(
            cls._old_parser_config[2][0],
            projects_dir=cls._old_parser_config[2][1],
            chats_dir=cls._old_parser_config[2][2],
        )
        opencode.configure(cls._old_parser_config[3])
        chatgpt.configure(cls._old_parser_config[4])
        cls._tmp.cleanup()

    def get(self, path: str):
        return http_get(self.port, path, timeout=5)

    def put_json(self, path: str, payload: dict):
        return _send_json(self.port, "PUT", path, payload)

    def post_json(self, path: str, payload: dict):
        return _send_json(self.port, "POST", path, payload)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def _user(uuid, parent, text, ts):
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": ts,
        "message": {"role": "user", "content": text},
    }


def _assistant(uuid, parent, text, ts):
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "timestamp": ts,
        "message": {
            "role": "assistant",
            "model": "claude-test",
            "content": [{"type": "text", "text": text}],
        },
    }


def _write_fixture_session(
    projects_dir: Path,
    session_id: str = "11111111-1111-1111-1111-111111111111",
    prompt: str = "hello world",
    additional_prompts: tuple[str, ...] = (),
    extra_records: tuple[dict, ...] = (),
) -> Path:
    """A minimal but valid Claude Code transcript so endpoints have real data."""
    proj = projects_dir / "-tmp-proj"
    proj.mkdir(parents=True, exist_ok=True)
    f = proj / f"{session_id}.jsonl"
    records = [
        {"type": "user", "timestamp": "2024-01-01T00:00:00Z", "cwd": "/tmp/proj",
         "message": {"role": "user", "content": prompt}},
        {"type": "assistant", "timestamp": "2024-01-01T00:00:01Z",
         "message": {"role": "assistant", "model": "claude-test",
                     "content": [{"type": "text", "text": "hi there"}]}},
    ]
    for i, extra_prompt in enumerate(additional_prompts, start=2):
        records.extend([
            {"type": "user", "timestamp": f"2024-01-01T00:00:{i:02d}Z",
             "message": {"role": "user", "content": extra_prompt}},
            {"type": "assistant", "timestamp": f"2024-01-01T00:00:{i + 1:02d}Z",
             "message": {"role": "assistant", "model": "claude-test",
                         "content": [{"type": "text", "text": "continued reply"}]}},
        ])
    records.extend(extra_records)
    f.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return f


def _write_cli_store(
    chats_dir: Path,
    session_id: str = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    cwd: str = "/Users/test/demo",
    *,
    title: str = "Store db session",
    user_text: str = "<user_query>\nhello from store db\n</user_query>",
    meta_extra: dict | None = None,
) -> Path:
    """Minimal Cursor CLI store.db with a tool call + result."""
    sess = chats_dir / "deadbeefcafebabe" / session_id
    sess.mkdir(parents=True)
    (sess / "meta.json").write_text(
        json.dumps({
            "schemaVersion": 1,
            "createdAtMs": 1_700_000_000_000,
            "updatedAtMs": 1_700_000_100_000,
            "hasConversation": True,
            "title": title,
            "cwd": cwd,
        }),
        encoding="utf-8",
    )
    db = sess / "store.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("CREATE TABLE blobs (id TEXT PRIMARY KEY, data BLOB)")
    meta = {
        "agentId": session_id,
        "latestRootBlobId": "0" * 64,
        "name": title,
        "mode": "default",
        "createdAt": 1_700_000_000_000,
        "lastUsedModel": "grok-test",
    }
    meta.update(meta_extra or {})
    conn.execute(
        "INSERT INTO meta(key, value) VALUES ('0', ?)",
        (json.dumps(meta).encode("utf-8").hex(),),
    )
    blobs = [
        ("a" * 64, json.dumps({
            "role": "user",
            "content": [{
                "type": "text",
                "text": user_text,
            }],
        }).encode()),
        ("b" * 64, json.dumps({
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Running a command."},
                {
                    "type": "tool-call",
                    "toolCallId": "call-1",
                    "toolName": "Shell",
                    "args": {"command": "echo hi", "description": "say hi"},
                },
            ],
        }).encode()),
        ("c" * 64, json.dumps({
            "role": "tool",
            "content": [{
                "type": "tool-result",
                "toolCallId": "call-1",
                "toolName": "Shell",
                "result": "Exit code: 0\n\nhi\n",
            }],
        }).encode()),
        ("d" * 64, json.dumps({
            "role": "assistant",
            "content": [{"type": "text", "text": "Done."}],
        }).encode()),
    ]
    for bid, data in blobs:
        conn.execute("INSERT INTO blobs(id, data) VALUES (?, ?)", (bid, data))
    conn.commit()
    conn.close()
    return db


def _write_cli_session(
    projects_dir: Path,
    session_id: str = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    project_slug: str = "Users-test-demo",
    user_text: str = "hello from cursor cli",
) -> Path:
    """Minimal Cursor CLI agent-transcripts JSONL under a fake projects dir."""
    session_dir = projects_dir / project_slug / "agent-transcripts" / session_id
    session_dir.mkdir(parents=True)
    path = session_dir / f"{session_id}.jsonl"
    records = [
        {
            "role": "user",
            "message": {
                "content": [{
                    "type": "text",
                    "text": (
                        f"<timestamp>Monday, Jan 1, 2024</timestamp>\n"
                        f"<user_query>\n{user_text}\n</user_query>"
                    ),
                }],
            },
        },
        {
            "role": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Looking into it."},
                    {
                        "type": "tool_use",
                        "name": "Shell",
                        "input": {
                            "command": "echo hi",
                            "description": "say hi",
                            "working_directory": "/Users/test/demo",
                        },
                    },
                ],
            },
        },
        {
            "role": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "StrReplace",
                        "input": {
                            "path": "/Users/test/demo/a.py",
                            "old_string": "x = 1",
                            "new_string": "x = 2",
                        },
                    },
                ],
            },
        },
    ]
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return path


def _write_guardian_sessions(codex_home: Path) -> tuple[Path, Path, Path]:
    sessions = codex_home / "sessions" / "2026" / "01" / "01"
    sessions.mkdir(parents=True)
    parent_id = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    guardian_id = "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb"
    parent = sessions / f"rollout-2026-01-01T00-00-00-{parent_id}.jsonl"
    guardian = sessions / f"rollout-2026-01-01T00-00-01-{guardian_id}.jsonl"
    image = sessions / "fixture.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    inline_image = "data:image/png;base64,iVBORw0KGgo="
    parent_records = [
        {"timestamp": "2026-01-01T00:00:00Z", "type": "session_meta",
         "payload": {"id": parent_id, "cwd": "/tmp/proj"}},
        {"timestamp": "2026-01-01T00:00:01Z", "type": "event_msg",
         "payload": {"type": "user_message", "message": "parent task"}},
        {"timestamp": "2026-01-01T00:00:01Z", "type": "response_item",
         "payload": {"type": "message", "role": "user", "content": [
             {"type": "input_text", "text": f'<image name=[Image #1] path="{image}">'},
             {"type": "input_image", "image_url": inline_image},
             {"type": "input_text", "text": "</image>"},
             {"type": "input_text", "text": "look at this"},
         ]}},
        {"timestamp": "2026-01-01T00:00:01Z", "type": "event_msg",
         "payload": {"type": "user_message", "message": "look at this",
                     "images": [], "local_images": [str(image)]}},
        {"timestamp": "2026-01-01T00:00:01Z", "type": "response_item",
         "payload": {"type": "message", "role": "user", "content": [
             {"type": "input_text", "text": '<image name=[Image #1] path="/missing.png">'},
             {"type": "input_image", "image_url": inline_image},
             {"type": "input_text", "text": "</image>"},
             {"type": "input_text", "text": "missing image"},
         ]}},
        {"timestamp": "2026-01-01T00:00:01Z", "type": "event_msg",
         "payload": {"type": "user_message", "message": "missing image",
                     "images": [], "local_images": ["/missing.png"]}},
        {"timestamp": "2026-01-01T00:00:02Z", "type": "event_msg",
         "payload": {"type": "task_started", "turn_id": "parent-turn",
                     "model_context_window": 300000, "collaboration_mode_kind": "default"}},
        {"timestamp": "2026-01-01T00:00:02Z", "type": "turn_context",
         "payload": {"turn_id": "parent-turn", "model": "codex-test", "effort": "medium",
                     "approval_policy": "on-request", "sandbox_policy": {"type": "workspace-write"}}},
        {"timestamp": "2026-01-01T00:00:03Z", "type": "event_msg",
         "payload": {"type": "user_message", "message": "metadata turn"}},
        {"timestamp": "2026-01-01T00:00:04Z", "type": "event_msg",
         "payload": {"type": "agent_message", "message": "metadata answer"}},
        {"timestamp": "2026-01-01T00:00:04Z", "type": "event_msg",
         "payload": {"type": "token_count", "info": {
             "model_context_window": 300000,
             "total_token_usage": {"input_tokens": 100, "output_tokens": 10},
         }}},
        {"timestamp": "2026-01-01T00:00:05Z", "type": "event_msg",
         "payload": {"type": "task_complete", "turn_id": "parent-turn",
                     "duration_ms": 3000, "time_to_first_token_ms": 700}},
        {"timestamp": "2026-01-01T00:00:06Z", "type": "compacted",
         "payload": {
             "window_number": 1,
             "previous_window_id": "window-old",
             "window_id": "window-new",
             "message": "",
             "replacement_history": [
                 {"type": "message", "role": "user", "content": [
                     {"type": "input_text", "text": "retained prompt"},
                 ]},
                 {"type": "compaction", "id": "cmp-test", "encrypted_content": "ciphertext"},
             ],
         }},
        {"timestamp": "2026-01-01T00:00:06Z", "type": "world_state",
         "payload": {"full": True, "state": {"environments": {"local": {"shell": "zsh"}}}}},
        {"timestamp": "2026-01-01T00:00:06Z", "type": "event_msg",
         "payload": {"type": "context_compacted"}},
    ]
    planned = {
        "command": ["/bin/zsh", "-lc", "python3 -m unittest tests.test_security"],
        "cwd": "/tmp/proj",
        "justification": "Run local tests?",
        "sandbox_permissions": "require_escalated",
        "tool": "exec_command",
    }
    guardian_records = [
        {"timestamp": "2026-01-01T00:00:02Z", "type": "session_meta",
         "payload": {
             "id": guardian_id, "parent_thread_id": parent_id,
             "thread_source": "subagent", "source": {"subagent": {"other": "guardian"}},
             "cwd": "/tmp/proj", "base_instructions": {"text": "Review actions."},
         }},
        {"timestamp": "2026-01-01T00:00:02Z", "type": "event_msg",
         "payload": {"type": "task_started", "turn_id": "turn-1",
                     "model_context_window": 200000,
                     "collaboration_mode_kind": "codex-auto-review"}},
        {"timestamp": "2026-01-01T00:00:02Z", "type": "turn_context",
         "payload": {"turn_id": "turn-1", "model": "guardian-test", "effort": "low",
                     "approval_policy": "never", "sandbox_policy": {"type": "read-only"}}},
        {"timestamp": "2026-01-01T00:00:03Z", "type": "event_msg",
         "payload": {"type": "user_message",
                     "message": "Review this action.\nPlanned action JSON:\n" + json.dumps(planned)}},
        {"timestamp": "2026-01-01T00:00:04Z", "type": "event_msg",
         "payload": {"type": "agent_message", "message": '{"outcome":"allow"}'}},
        {"timestamp": "2026-01-01T00:00:04Z", "type": "event_msg",
         "payload": {"type": "token_count", "info": {
             "model_context_window": 200000,
             "total_token_usage": {"input_tokens": 1200, "output_tokens": 20},
         }}},
        {"timestamp": "2026-01-01T00:00:05Z", "type": "event_msg",
         "payload": {"type": "task_complete", "turn_id": "turn-1",
                     "duration_ms": 2000, "time_to_first_token_ms": 500}},
    ]
    parent.write_text("\n".join(json.dumps(r) for r in parent_records) + "\n")
    guardian.write_text("\n".join(json.dumps(r) for r in guardian_records) + "\n")
    return parent, guardian, image


def _write_opencode_db(db_path: Path) -> tuple[str, str]:
    """A minimal opencode.db holding a parent session and its `task` sub-agent.

    Covers every part type the parser has a branch for — text, reasoning, tool
    (completed / error / pending), file, synthetic text, subtask, compaction,
    retry, patch and step-finish — so schema conformance is exercised end to
    end. Returns the (parent, sub-agent) session ids.
    """
    parent_id = "ses_parent000000000000000000"
    child_id = "ses_child0000000000000000000"
    model = json.dumps({"providerID": "openrouter", "modelID": "x-ai/grok-test"})
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE session (
            id TEXT PRIMARY KEY, project_id TEXT NOT NULL, parent_id TEXT,
            slug TEXT NOT NULL, directory TEXT NOT NULL, title TEXT NOT NULL,
            version TEXT NOT NULL, cost REAL DEFAULT 0 NOT NULL, agent TEXT,
            model TEXT, time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL
        );
        CREATE TABLE message (
            id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
            time_created INTEGER NOT NULL, data TEXT NOT NULL
        );
        CREATE TABLE part (
            id TEXT PRIMARY KEY, message_id TEXT NOT NULL, session_id TEXT NOT NULL,
            data TEXT NOT NULL
        );
        """
    )
    conn.executemany(
        "INSERT INTO session (id, project_id, parent_id, slug, directory, title, "
        "version, cost, agent, model, time_created, time_updated) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (parent_id, "proj1", None, "brisk-otter", "/tmp/proj", "Fixture session",
             "1.18.18", 0.25, "build", model, 1_700_000_000_000, 1_700_000_100_000),
            # opencode leaves this placeholder title until the model names the
            # session; the parser must fall back to the first user prompt.
            (child_id, "proj1", parent_id, "tidy-cactus", "/tmp/proj",
             "New session - 2026-01-01T00:00:00.000Z", "1.18.18", 0.05, "explore",
             model, 1_700_000_010_000, 1_700_000_050_000),
        ],
    )

    def message(mid, sid, role, created, **extra):
        data = {"role": role, "time": {"created": created}, **extra}
        if role == "assistant":
            data.setdefault("modelID", "x-ai/grok-test")
            data.setdefault("providerID", "openrouter")
            data.setdefault("tokens", {"input": 100, "output": 20, "reasoning": 5,
                                       "cache": {"read": 10, "write": 0}})
        conn.execute(
            "INSERT INTO message (id, session_id, time_created, data) VALUES (?,?,?,?)",
            (mid, sid, created, json.dumps(data)),
        )

    def part(pid, mid, sid, data):
        conn.execute(
            "INSERT INTO part (id, message_id, session_id, data) VALUES (?,?,?,?)",
            (pid, mid, sid, json.dumps(data)),
        )

    # --- parent session -----------------------------------------------------
    message("msg_p1", parent_id, "user", 1_700_000_000_100, agent="build")
    part("prt_p1a", "msg_p1", parent_id, {"type": "text", "text": "look at the repo"})
    part("prt_p1b", "msg_p1", parent_id, {
        "type": "file", "mime": "image/png", "filename": "shot.png",
        "url": "data:image/png;base64,iVBORw0KGgo=",
    })
    part("prt_p1c", "msg_p1", parent_id, {
        "type": "text", "synthetic": True, "text": "<task>background result</task>"})

    message("msg_p2", parent_id, "assistant", 1_700_000_000_200,
            cost=0.2, finish="tool-calls", agent="build", variant="high")
    part("prt_p2a", "msg_p2", parent_id, {"type": "step-start", "snapshot": "abc123"})
    # Providers hang a token-by-token `reasoning_details` transcript off parts;
    # it is many times the size of the text and must never reach the viewer.
    # …and, for models that return one, an opaque `reasoning.encrypted` blob
    # that makes the visible text a summary rather than the real reasoning.
    token_noise = {"openrouter": {"reasoning_details": [
        {"type": "reasoning.summary", "summary": word, "index": 0}
        for word in ("Need", " to", " look", " around", " first", ".")
    ] + [
        {"type": "reasoning.encrypted", "id": "rs_fixture", "index": 1,
         "format": "xai-responses-v1", "data": "b3BhcXVl"},
    ]}}
    part("prt_p2b", "msg_p2", parent_id, {
        "type": "reasoning", "text": "Need to look around first.",
        "time": {"start": 1_700_000_000_250, "end": 1_700_000_000_290},
        "metadata": token_noise})
    part("prt_p2c", "msg_p2", parent_id, {
        "type": "text", "text": "Looking around.",
        "time": {"start": 1_700_000_000_295, "end": 1_700_000_000_299}})
    part("prt_p2d", "msg_p2", parent_id, {
        "type": "tool", "tool": "read", "callID": "call-1",
        "state": {"status": "completed", "title": "main.py",
                  "input": {"filePath": "/tmp/proj/main.py", "offset": 1, "limit": 20},
                  "output": "<path>/tmp/proj/main.py</path>",
                  "metadata": {"preview": "1: x = 1", "truncated": False},
                  "time": {"start": 1_700_000_000_300, "end": 1_700_000_000_350}},
        "metadata": token_noise})
    part("prt_p2e", "msg_p2", parent_id, {
        "type": "tool", "tool": "edit", "callID": "call-2",
        "state": {"status": "error", "error": "oldString not found",
                  "input": {"filePath": "/tmp/proj/main.py", "oldString": "a",
                            "newString": "b", "replaceAll": True},
                  "time": {"start": 1_700_000_000_360, "end": 1_700_000_000_370}}})
    part("prt_p2f", "msg_p2", parent_id, {
        "type": "tool", "tool": "task", "callID": "call-3",
        "state": {"status": "completed", "title": "Explore repo",
                  "input": {"description": "Explore repo", "subagent_type": "explore",
                            "prompt": "look at everything"},
                  "output": f'<task id="{child_id}" state="completed">done</task>',
                  "metadata": {"parentSessionId": parent_id, "sessionId": child_id},
                  "time": {"start": 1_700_000_000_380, "end": 1_700_000_000_390}}})
    part("prt_p2g", "msg_p2", parent_id, {
        "type": "tool", "tool": "grep", "callID": "call-4",
        "state": {"status": "pending", "input": {}, "raw": ""}})
    part("prt_p2h", "msg_p2", parent_id, {
        "type": "patch", "hash": "deadbeef", "files": ["/tmp/proj/main.py"]})
    part("prt_p2i", "msg_p2", parent_id, {
        "type": "retry", "attempt": 1, "time": {"created": 1_700_000_000_150},
        "error": {"name": "APIError", "data": {"message": "429 slow down"}}})
    part("prt_p2j", "msg_p2", parent_id, {
        "type": "step-finish", "reason": "tool-calls", "cost": 0.2,
        "tokens": {"input": 100, "output": 20, "reasoning": 5,
                   "cache": {"read": 10, "write": 0}}})

    message("msg_p3", parent_id, "user", 1_700_000_000_300, agent="build")
    part("prt_p3a", "msg_p3", parent_id, {
        "type": "subtask", "prompt": "review the diff", "description": "Review",
        "agent": "plan"})
    part("prt_p3b", "msg_p3", parent_id, {"type": "compaction", "auto": True})

    # An assistant turn that only failed: no blocks, just the error notice.
    message("msg_p4", parent_id, "assistant", 1_700_000_000_400,
            error={"name": "ProviderAuthError",
                   "data": {"message": "Missing Authentication header"}})

    # --- sub-agent session --------------------------------------------------
    message("msg_c1", child_id, "user", 1_700_000_010_100, agent="explore")
    part("prt_c1a", "msg_c1", child_id, {"type": "text", "text": "look at everything"})
    message("msg_c2", child_id, "assistant", 1_700_000_010_200,
            cost=0.05, finish="stop", agent="explore")
    part("prt_c2a", "msg_c2", child_id, {"type": "text", "text": "Here is what I found."})

    conn.commit()
    conn.close()
    return parent_id, child_id


# A 1×1 transparent PNG: small enough to inline, real enough to sniff.
TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)


def _chatgpt_message(node_id, role, content, create_time, **extra):
    message = {
        "id": node_id,
        "author": {"role": role, "name": extra.pop("author_name", None)},
        "create_time": create_time,
        "content": content,
        "recipient": extra.pop("recipient", "all"),
        "metadata": extra.pop("metadata", {}),
    }
    message.update(extra)
    return message


def chatgpt_conversation(conv_id: str = "c0ffee00-0000-4000-8000-000000000001",
                         title: str = "Netboot planning", update_time: float = 1767225900.0) -> dict:
    """One conversation exercising what chatgpt_parser handles: an edited
    prompt (so an abandoned branch), hidden and custom-instruction messages,
    thoughts, a citation marker, a tool round trip, and an uploaded image."""
    t0 = 1767225600.0  # 2026-01-01T00:00:00Z
    nodes = [
        ("root", None, None),
        ("sys", "root", _chatgpt_message(
            "sys", "system", {"content_type": "text", "parts": [""]}, None,
            metadata={"is_visually_hidden_from_conversation": True})),
        ("ctx", "sys", _chatgpt_message(
            "ctx", "user", {"content_type": "user_editable_context",
                            "user_profile": "I run a home lab.", "user_instructions": "Be terse."},
            None, metadata={"is_visually_hidden_from_conversation": False})),
        ("u1-old", "ctx", _chatgpt_message(
            "u1-old", "user", {"content_type": "text", "parts": ["abandonedword first draft"]}, t0)),
        ("a1-old", "u1-old", _chatgpt_message(
            "a1-old", "assistant", {"content_type": "text", "parts": ["old reply"]}, t0 + 5,
            metadata={"model_slug": "gpt-4o"})),
        ("u1", "ctx", _chatgpt_message(
            "u1", "user", {"content_type": "text", "parts": ["How do I set up netboot.xyz?"]}, t0 + 60)),
        ("th", "u1", _chatgpt_message(
            "th", "assistant", {"content_type": "thoughts", "thoughts": [
                {"summary": "Planning", "content": "Consider dnsmasq proxy mode.", "chunks": [], "finished": True}]},
            t0 + 61, metadata={"model_slug": "gpt-5-thinking"})),
        ("rr", "th", _chatgpt_message(
            "rr", "assistant", {"content_type": "reasoning_recap", "content": "Thought for 3s"}, t0 + 62,
            metadata={"model_slug": "gpt-5-thinking"})),
        ("call", "rr", _chatgpt_message(
            "call", "assistant", {"content_type": "code", "language": "unknown", "text": "print(2 + 2)"},
            t0 + 63, recipient="python", metadata={"model_slug": "gpt-5-thinking"})),
        ("out", "call", _chatgpt_message(
            "out", "tool", {"content_type": "execution_output", "text": "4"}, t0 + 64, author_name="python")),
        ("a1", "out", _chatgpt_message(
            "a1", "assistant", {"content_type": "text", "parts": [
                "Use proxy DHCP citeturn0search0 and see "
                "urlthe docshttps://netboot.xyz/docs."]},
            t0 + 65, metadata={"model_slug": "gpt-5-thinking", "content_references": [
                {"matched_text": "citeturn0search0", "type": "grouped_webpages",
                 "items": [{"title": "dnsmasq man page", "url": "https://example.test/dnsmasq"}]}]})),
        ("u2", "a1", _chatgpt_message(
            "u2", "user", {"content_type": "multimodal_text", "parts": [
                {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-Abc123",
                 "size_bytes": len(TINY_PNG), "width": 1, "height": 1},
                "What is in this screenshot?"]},
            t0 + 120, metadata={"attachments": [
                {"id": "file-Abc123", "name": "screenshot.png", "mime_type": "image/png"}]})),
        ("a2", "u2", _chatgpt_message(
            "a2", "assistant", {"content_type": "text", "parts": ["A single transparent pixel."]}, t0 + 125,
            metadata={"model_slug": "gpt-5-thinking"})),
    ]
    mapping = {node_id: {"id": node_id, "message": msg, "parent": parent, "children": []}
               for node_id, parent, msg in nodes}
    for node_id, parent, _msg in nodes:
        if parent:
            mapping[parent]["children"].append(node_id)
    return {
        "id": conv_id, "conversation_id": conv_id, "title": title,
        "create_time": t0, "update_time": update_time,
        "default_model_slug": "gpt-5-thinking", "current_node": "a2", "mapping": mapping,
        "is_archived": False,
    }


def _write_chatgpt_export(path: Path, conversations: list[dict] | None = None, *,
                          sharded: bool = False, extracted: bool = False) -> Path:
    """A ChatGPT data export at `path` (a .zip, or a directory if `extracted`),
    with the conversation JSON either whole or split into shards."""
    import zipfile

    conversations = conversations if conversations is not None else [chatgpt_conversation()]
    if sharded:
        files = {f"conversations-{i:03d}.json": json.dumps([c]) for i, c in enumerate(conversations)}
    else:
        files = {"conversations.json": json.dumps(conversations)}
    files["user.json"] = json.dumps({"id": "user-test"})
    blobs = {"file-Abc123-screenshot.png": TINY_PNG}
    if extracted:
        path.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (path / name).write_text(text, encoding="utf-8")
        for name, raw in blobs.items():
            (path / name).write_bytes(raw)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        for name, text in files.items():
            zf.writestr(name, text)
        for name, raw in blobs.items():
            zf.writestr(name, raw)
    return path
