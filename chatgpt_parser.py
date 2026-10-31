#!/usr/bin/env python3
"""Parser for ChatGPT data exports.

ChatGPT's "Export data" (Settings → Data controls) mails a ``.zip`` holding the
whole account history. The viewer reads that archive in place — read-only, never
extracted — or an already-extracted copy of it. Sessions are addressed by the
synthetic id ``chatgpt:<conversation_id>``, as opencode's database rows are.

Export layout (2023–2026):

- ``conversations.json``, or in newer exports ``conversations-000.json``,
  ``conversations-001.json``, … — a list of conversations. Each has a ``title``,
  ``create_time``/``update_time`` (epoch seconds), ``default_model_slug``,
  ``current_node`` and a ``mapping`` of node id → ``{message, parent, children}``.
  The mapping is a tree: editing a prompt or regenerating a reply forks it, and
  ``current_node`` is the leaf of the branch the user last saw.
- Each ``message`` has ``author.role`` (user / assistant / tool / system),
  ``recipient`` (``"all"``, or the tool an assistant message is addressed to),
  ``content`` discriminated on ``content_type`` (text, multimodal_text, thoughts,
  reasoning_recap, code, execution_output, user_editable_context, …), and
  ``metadata`` (``model_slug``, ``attachments``, ``content_references``, …).
- Uploaded and generated images ride along as ``file-<id>.dat`` /
  ``file_<hex>.dat`` (older exports keep the original name after the id),
  referenced from a message part as ``file-service://file-<id>`` or
  ``sediment://file_<hex>``.

Tool calls are assistant messages addressed to a tool, answered by a following
``tool`` message; they fold into the assistant turn as ``tool_use`` blocks, so
this parser emits the *block* event shape, like claude_parser and opencode_parser.
Paths not on the active branch become ``branch`` events at the fork.

Unlike the other sources there is no fixed on-disk location, so exports are found
via ``configure()``: each path may be an export ``.zip``, an extracted export
directory, or a directory holding any number of either (one level deep).
"""

from __future__ import annotations

import base64
import json
import re
import threading
import zipfile
from pathlib import Path, PurePosixPath

import common

# Drop export zips (or extracted exports) here to have them picked up.
DEFAULT_EXPORTS_DIR = Path.home() / ".config" / "cc_transcript_viewer" / "chatgpt"
EXPORT_PATHS: list[Path] = [DEFAULT_EXPORTS_DIR]

SESSION_SCHEME = "chatgpt:"

# Present for the server's cache persistence, which walks every parser. The
# export is parsed whole whenever it changes, so there is nothing to persist.
SUMMARY_CACHE = common.SummaryCache()

# Images are inlined as data: URIs when a session is opened; past this size a
# placeholder note is shown instead.
MAX_IMAGE_BYTES = 8 * 1024 * 1024

_CONVERSATION_FILE_RE = re.compile(r"^conversations(?:-\d+)?\.json$")
_ASSET_ID_RE = re.compile(r"^(file[-_][A-Za-z0-9]+)")
_WEB_TOOL_RE = re.compile(r"web|browser|search", re.I)

_LOCK = threading.Lock()
# (identity, {conversation_id: session dict}, [summary dicts])
_STATE: tuple | None = None


def configure(paths=None) -> None:
    """Point the parser at export zips/directories (a path or a list of them)."""
    global EXPORT_PATHS, _STATE
    if paths is None:
        paths = [DEFAULT_EXPORTS_DIR]
    elif isinstance(paths, (str, Path)):
        paths = [paths]
    EXPORT_PATHS = [Path(p).expanduser() for p in paths]
    with _LOCK:
        _STATE = None


# ---------------------------------------------------------------------------
# Reading exports (zip or directory)
# ---------------------------------------------------------------------------
class Export:
    """One ChatGPT export, as a zip archive or an extracted directory."""

    def __init__(self, path: Path, conversation_files: list[str], prefix: str = ""):
        self.path = path
        self.is_zip = path.is_file()
        self.conversation_files = conversation_files
        self.prefix = prefix  # folder inside the zip that holds the export, if any

    def identity(self) -> tuple:
        if self.is_zip:
            return (str(self.path),) + tuple(common.file_identity(self.path) or ())
        idents = [common.file_identity(self.path / name) for name in self.conversation_files]
        return (str(self.path),) + tuple(x for ident in idents for x in (ident or (0, 0)))

    def read(self, name: str) -> bytes:
        if self.is_zip:
            with zipfile.ZipFile(self.path) as zf:
                return zf.read(self.prefix + name)
        return (self.path / name).read_bytes()

    def names(self) -> list[str]:
        if self.is_zip:
            with zipfile.ZipFile(self.path) as zf:
                return [
                    n[len(self.prefix):] for n in zf.namelist()
                    if n.startswith(self.prefix) and not n.endswith("/")
                ]
        return [
            p.relative_to(self.path).as_posix()
            for p in self.path.rglob("*") if p.is_file()
        ]

    def asset_index(self) -> dict[str, str]:
        """Asset id (``file-…`` / ``file_…``) → member name, for every bundled file."""
        index = {}
        for name in self.names():
            m = _ASSET_ID_RE.match(PurePosixPath(name).name)
            if m:
                index.setdefault(m.group(1), name)
        return index


def _sort_conversation_files(names) -> list[str]:
    return sorted(names, key=lambda n: (n != "conversations.json", n))


def open_export(path: Path) -> Export | None:
    """An Export for `path` when it is a ChatGPT export zip or directory."""
    try:
        if path.is_file() and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                members = zf.namelist()
            by_prefix: dict[str, list[str]] = {}
            for member in members:
                posix = PurePosixPath(member)
                if _CONVERSATION_FILE_RE.match(posix.name):
                    prefix = "" if str(posix.parent) == "." else str(posix.parent) + "/"
                    by_prefix.setdefault(prefix, []).append(posix.name)
            if not by_prefix:
                return None
            prefix = min(by_prefix, key=len)  # the export root, not a nested copy
            return Export(path, _sort_conversation_files(by_prefix[prefix]), prefix)
        if path.is_dir():
            names = [p.name for p in path.iterdir() if p.is_file() and _CONVERSATION_FILE_RE.match(p.name)]
            if names:
                return Export(path, _sort_conversation_files(names))
    except (OSError, zipfile.BadZipFile):
        return None
    return None


def discover_exports(paths=None) -> list[Export]:
    """Every export reachable from the configured paths (one level deep)."""
    found: list[Export] = []
    seen: set[str] = set()
    for root in EXPORT_PATHS if paths is None else paths:
        root = Path(root).expanduser()
        candidates = [root]
        if root.is_dir() and open_export(root) is None:
            try:
                candidates = sorted(root.iterdir())
            except OSError:
                candidates = []
        for candidate in candidates:
            export = open_export(candidate)
            if export is not None and str(export.path) not in seen:
                seen.add(str(export.path))
                found.append(export)
    return found


def load_conversations(export: Export) -> list[dict]:
    conversations: list[dict] = []
    for name in export.conversation_files:
        try:
            data = json.loads(export.read(name))
        except (OSError, KeyError, ValueError, zipfile.BadZipFile):
            continue
        if isinstance(data, list):
            conversations.extend(c for c in data if isinstance(c, dict))
    return conversations


# ---------------------------------------------------------------------------
# Message → event conversion
# ---------------------------------------------------------------------------
def _iso(seconds) -> str | None:
    if not isinstance(seconds, (int, float)) or not seconds:
        return None
    return common.iso_from_ms(seconds * 1000) or None


# ChatGPT marks citations and rich entities inline with private-use characters:
# U+E200 opens a marker, U+E202 separates its fields, U+E201 closes it, e.g.
# "citeturn0search3". Older exports used 【6†source】 instead.
_MARKER_RE = re.compile("([^]*)")
_OLD_CITE_RE = re.compile("【[^【】]{0,80}】")
_STRAY_PUA_RE = re.compile("[-]")


def _reference_links(ref: dict) -> str:
    links = []
    for item in ref.get("items") or []:
        if isinstance(item, dict) and item.get("url"):
            links.append(_link(item))
    return " (" + "; ".join(links) + ")" if links else ""


def _marker_text(body: str) -> str:
    kind, *fields = body.split("")
    fields = [f for f in fields if f]
    if kind in ("url", "link_title") and fields:
        label = fields[0]
        url = next((f for f in fields[1:] if f.startswith(("http://", "https://"))), "")
        return f"[{label}]({url})" if url else label
    if kind == "summary" and fields:
        return fields[0]
    if kind in ("entity", "product_entity", "entity_metadata") and fields:
        try:
            value = json.loads(fields[0])
        except ValueError:
            return ""
        names = [v for v in value if isinstance(v, str)] if isinstance(value, list) else []
        if kind == "entity" and len(names) >= 2:
            return names[1]
        return names[-1] if names else ""
    return ""  # cite, filecite, navlist, image_group, map, … carry no readable text


def _clean_url(url: str) -> str:
    """Drop the `utm_source=chatgpt.com` ChatGPT appends to every cited link."""
    url = re.sub(r"([?&])utm_source=chatgpt\.com(&|$)", lambda m: m.group(1) if m.group(2) else "", url)
    return url.rstrip("?&")


def _citation_index(references) -> dict[str, dict]:
    """Citation key ('turn0search3') -> cited item, from each item's `refs`.

    Newer exports leave every reference's `matched_text` empty, so this is the
    only reliable way back from an inline cite marker to the page it names.
    (The references are *not* in marker order, so position can't be used.)
    """
    index: dict[str, dict] = {}
    for ref in references or []:
        if not isinstance(ref, dict):
            continue
        for item in ref.get("items") or []:
            if not isinstance(item, dict) or not item.get("url"):
                continue
            for r in item.get("refs") or []:
                if isinstance(r, dict) and r.get("ref_type") is not None:
                    key = f"turn{r.get('turn_index')}{r.get('ref_type')}{r.get('ref_index')}"
                    index.setdefault(key, item)
    return index


def _link(item: dict) -> str:
    url = _clean_url(item["url"])
    title = (item.get("title") or item.get("attribution") or url).replace("]", "").replace("\n", " ")
    return f"[{title}]({url})"


def clean_text(text: str, references=None) -> str:
    """Replace ChatGPT's inline citation/entity markers with readable Markdown."""
    # Any marker character, not just U+E200: attribution highlights wrap a
    # span in U+E203\u2026U+E204 (+ U+E206) with no opening U+E200 at all.
    if not text or (not _STRAY_PUA_RE.search(text) and "\u3010" not in text):
        return text
    by_marker = {}
    for ref in references or []:
        if isinstance(ref, dict) and ref.get("matched_text"):
            by_marker.setdefault(ref["matched_text"], ref)
    by_key = _citation_index(references)

    def replace(match: re.Match) -> str:
        ref = by_marker.get(match.group(0))
        links = _reference_links(ref) if ref else ""
        kind, *keys = match.group(1).split("\ue202")
        if not links and kind in ("cite", "filecite"):
            items = []
            for key in keys:
                item = by_key.get(key)
                if item is not None and item not in items:
                    items.append(item)
            links = " (" + "; ".join(_link(i) for i in items) + ")" if items else ""
        if links and match.start() and match.string[match.start() - 1].isspace():
            links = links.lstrip()  # "text cite…" already has its space
        return links or _marker_text(match.group(1))

    def replace_old(match: re.Match) -> str:
        ref = by_marker.get(match.group(0))
        return _reference_links(ref) if ref else ""

    text = _MARKER_RE.sub(replace, text)
    text = _OLD_CITE_RE.sub(replace_old, text)
    return _STRAY_PUA_RE.sub("", text)


def _asset_id(pointer) -> str:
    """'file-service://file-Abc' / 'sediment://file_00ab…' → 'file-Abc' / 'file_00ab…'."""
    if not isinstance(pointer, str):
        return ""
    return pointer.rsplit("/", 1)[-1]


def _image_block(part: dict, attachments: dict) -> dict:
    asset = _asset_id(part.get("asset_pointer"))
    att = attachments.get(asset) or {}
    block = {"type": "image", "data_uri": "", "asset": asset}
    if att.get("name"):
        block["name"] = att["name"]
    if att.get("mime_type"):
        block["mime_type"] = att["mime_type"]
    if part.get("size_bytes"):
        block["bytes"] = part["size_bytes"]
    return block


def _content_text(content: dict) -> str:
    """Best-effort plain text of any content type (used for tools and fallbacks)."""
    for key in ("text", "result", "content"):
        value = content.get(key)
        if isinstance(value, str) and value.strip():
            return value
    parts = []
    for part in content.get("parts") or []:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "\n".join(p for p in parts if p)


def _message_blocks(message: dict) -> list[dict]:
    """Text/image/thinking blocks for a user or assistant message addressed to all."""
    content = message.get("content") or {}
    ctype = content.get("content_type")
    meta = message.get("metadata") or {}
    refs = meta.get("content_references")
    attachments = {
        a.get("id"): a for a in meta.get("attachments") or [] if isinstance(a, dict) and a.get("id")
    }

    if ctype == "thoughts":
        chunks = []
        for thought in content.get("thoughts") or []:
            if not isinstance(thought, dict):
                continue
            summary = (thought.get("summary") or "").strip()
            body = (thought.get("content") or "").strip()
            chunks.append(f"**{summary}**\n\n{body}" if summary and body else summary or body)
        text = "\n\n".join(c for c in chunks if c)
        return [{"type": "thinking", "text": text}] if text else []
    if ctype == "reasoning_recap":
        return []  # "Thought for 12 seconds" — the thoughts themselves are already shown
    if ctype in ("model_editable_context", "user_editable_context"):
        return []  # memory / custom instructions; handled as instructions where relevant
    if ctype == "code":
        lang = content.get("language") or ""
        if lang == "unknown":
            lang = ""
        text = content.get("text") or ""
        return [{"type": "text", "text": f"```{lang}\n{text}\n```"}] if text.strip() else []

    blocks: list[dict] = []
    for part in content.get("parts") or []:
        if isinstance(part, str):
            text = clean_text(part, refs)
            if text.strip():
                blocks.append({"type": "text", "text": text})
        elif isinstance(part, dict):
            ptype = part.get("content_type")
            if ptype == "image_asset_pointer":
                blocks.append(_image_block(part, attachments))
            elif ptype == "audio_transcription" and (part.get("text") or "").strip():
                blocks.append({"type": "text", "text": part["text"].strip()})
    if not blocks and ctype not in ("text", "multimodal_text"):
        text = _content_text(content)
        if text.strip():
            blocks.append({"type": "text", "text": text})

    shown = {b.get("asset") for b in blocks if b["type"] == "image"}
    files = [a.get("name") for aid, a in attachments.items() if aid not in shown and a.get("name")]
    if files:
        blocks.append({"type": "text", "text": "📎 " + ", ".join(files)})
    return blocks


# Reference types that name the web pages a reply cites.
_CITED_PAGE_TYPES = {"grouped_webpages", "grouped_webpages_model_predicted_fallback", "webpage", "webpage_extended"}


def _sources_block(meta: dict) -> dict | None:
    """A "Sources" text block: every page the reply cites, once each.

    ChatGPT's own source list (`sources_footnote`) comes first, then any page
    an inline citation names that the footnote missed — inline markers can't
    always be resolved to a position, but the page they cite is never lost.
    """
    pages: list[dict] = []
    seen: set[str] = set()

    def add(item) -> None:
        if not isinstance(item, dict) or not item.get("url"):
            return
        url = _clean_url(item["url"])
        if url not in seen:
            seen.add(url)
            pages.append(item)

    refs = [r for r in meta.get("content_references") or [] if isinstance(r, dict)]
    for ref in refs:
        if ref.get("type") == "sources_footnote":
            for source in ref.get("sources") or []:
                add(source)
    for ref in refs:
        if ref.get("type") in _CITED_PAGE_TYPES:
            add(ref)
            for item in ref.get("items") or []:
                add(item)
    if not pages:
        return None
    lines = []
    for page in pages:
        site = page.get("attribution") or ""
        lines.append("- " + _link(page) + (f" · {site}" if site and site != page.get("title") else ""))
    return {"type": "text", "text": "**Sources**\n\n" + "\n".join(lines)}


def _search_block(meta: dict) -> dict | None:
    """The pages a web lookup retrieved (`search_result_groups`), as a tool call.

    The export keeps the results but not the query, so the input lists the
    sites searched; the result lists each page as title, URL, and snippet.
    """
    groups = [g for g in meta.get("search_result_groups") or [] if isinstance(g, dict)]
    entries, lines, seen = [], [], set()
    for group in groups:
        for entry in group.get("entries") or []:
            if isinstance(entry, dict) and entry.get("url"):
                url = _clean_url(entry["url"])
                if url in seen:
                    continue
                seen.add(url)
                entries.append(entry)
                snippet = " ".join((entry.get("snippet") or "").split())
                lines.append("\n".join(x for x in (
                    entry.get("title") or url, url, snippet[:300] + ("…" if len(snippet) > 300 else ""),
                ) if x))
    if not entries:
        return None
    domains = []
    for group in groups:
        if group.get("domain") and group["domain"] not in domains:
            domains.append(group["domain"])
    return {
        "type": "tool_use",
        "name": "web_search",
        "input": {"sites": domains},
        "result": {"text": f"{len(entries)} page{'s' if len(entries) != 1 else ''} retrieved\n\n"
                           + "\n\n".join(lines), "images": []},
    }


def _instructions_event(message: dict, ts) -> dict | None:
    content = message.get("content") or {}
    if content.get("content_type") == "user_editable_context":
        parts = [
            (content.get("user_profile") or "").strip(),
            (content.get("user_instructions") or "").strip(),
        ]
        text = "\n\n".join(p for p in parts if p)
        label = "Custom instructions"
    else:
        text = _content_text(content).strip()
        label = "System message"
    if not text:
        return None
    return {"kind": "instructions", "role": "system", "label": label, "text": text, "ts": ts}


class _EventBuilder:
    """Accumulates one linear message chain into events, merging consecutive
    assistant/tool messages into a single block-shaped assistant turn."""

    def __init__(self):
        self.events: list[dict] = []

    def _assistant(self, ts, model) -> dict:
        last = self.events[-1] if self.events else None
        if last and last["kind"] == "assistant":
            if model and not last.get("model"):
                last["model"] = model
            return last
        ev = {"kind": "assistant", "blocks": [], "ts": ts}
        if model:
            ev["model"] = model
        self.events.append(ev)
        return ev

    def add(self, message) -> None:
        if not isinstance(message, dict):
            return
        meta = message.get("metadata") or {}
        if meta.get("is_visually_hidden_from_conversation"):
            return
        role = (message.get("author") or {}).get("role")
        ts = _iso(message.get("create_time"))
        recipient = message.get("recipient")
        content = message.get("content") or {}

        if role == "system" or (role == "user" and content.get("content_type") == "user_editable_context"):
            ev = _instructions_event(message, ts)
            if ev:
                self.events.append(ev)
        elif role == "user":
            blocks = _message_blocks(message)
            if blocks:
                self.events.append({"kind": "user", "blocks": blocks, "ts": ts})
        elif role == "assistant" and recipient not in (None, "all"):
            text = _content_text(content)
            key = "code" if content.get("content_type") == "code" else "text"
            block = {"type": "tool_use", "name": recipient, "input": {key: text} if text else {}, "result": None}
            self._assistant(ts, meta.get("model_slug"))["blocks"].append(block)
        elif role == "assistant":
            # Retrieval happens before the text that uses it; the cited
            # sources follow the reply (only if there is one to cite from).
            blocks = _message_blocks(message)
            search = _search_block(meta)
            sources = _sources_block(meta) if any(b["type"] == "text" for b in blocks) else None
            blocks = ([search] if search else []) + blocks + ([sources] if sources else [])
            if blocks:
                self._assistant(ts, meta.get("model_slug"))["blocks"].extend(blocks)
        elif role == "tool":
            self._tool_result(message, ts)

    def _tool_result(self, message: dict, ts) -> None:
        name = (message.get("author") or {}).get("name") or "tool"
        content = message.get("content") or {}
        text = clean_text(_content_text(content), (message.get("metadata") or {}).get("content_references"))
        images = [
            _image_block(p, {}) for p in content.get("parts") or []
            if isinstance(p, dict) and p.get("content_type") == "image_asset_pointer"
        ]
        if not text.strip() and not images:
            return
        turn = self._assistant(ts, None)
        pending = next(
            (b for b in reversed(turn["blocks"]) if b["type"] == "tool_use" and b.get("result") is None),
            None,
        )
        if pending is None:
            pending = {"type": "tool_use", "name": name, "input": {}, "result": None}
            turn["blocks"].append(pending)
        pending["result"] = {"text": text, "images": []}
        # Generated images (DALL·E and friends) render as images in the turn.
        turn["blocks"].extend(images)


def _active_path(mapping: dict, leaf) -> list[str]:
    path, seen = [], set()
    node_id = leaf
    while node_id and node_id in mapping and node_id not in seen:
        seen.add(node_id)
        path.append(node_id)
        node_id = mapping[node_id].get("parent")
    return path[::-1]


def _latest_chain(mapping: dict, start: str) -> list[str]:
    """`start` and its most recent descendants (last child at every fork)."""
    chain, seen = [], set()
    node_id = start
    while node_id and node_id in mapping and node_id not in seen:
        seen.add(node_id)
        chain.append(node_id)
        children = [c for c in mapping[node_id].get("children") or [] if c in mapping]
        node_id = children[-1] if children else None
    return chain


def conversation_events(conv: dict) -> list[dict]:
    mapping = conv.get("mapping") or {}
    leaf = conv.get("current_node")
    if leaf not in mapping:
        # No recorded position: follow the newest branch from the root.
        roots = [k for k, v in mapping.items() if not v.get("parent")]
        chain = _latest_chain(mapping, roots[0]) if roots else []
        leaf = chain[-1] if chain else None
    path = _active_path(mapping, leaf)
    on_path = set(path)

    builder = _EventBuilder()
    for node_id in path:
        node = mapping[node_id]
        builder.add(node.get("message"))
        alternatives = [
            c for c in node.get("children") or [] if c in mapping and c not in on_path
        ]
        groups = []
        for alt in alternatives:
            sub = _EventBuilder()
            for alt_id in _latest_chain(mapping, alt):
                sub.add(mapping[alt_id].get("message"))
            if sub.events:
                groups.append(sub.events)
        if groups:
            builder.events.append({"kind": "branch", "groups": groups, "count": len(groups)})
    return builder.events


def _walk_blocks(events):
    for ev in events:
        if ev.get("kind") == "branch":
            for group in ev.get("groups") or []:
                yield from _walk_blocks(group)
        else:
            for b in ev.get("blocks") or []:
                yield ev, b


def _first_user_text(events) -> str:
    for ev in events:
        if ev.get("kind") == "user":
            for b in ev["blocks"]:
                if b["type"] == "text":
                    return b["text"]
    return ""


def build_session(conv: dict, export_index: int = 0) -> dict | None:
    conv_id = conv.get("conversation_id") or conv.get("id")
    if not isinstance(conv_id, str) or not conv_id:
        return None
    events = conversation_events(conv)
    models: list[str] = []
    for ev in events:
        if ev.get("model") and ev["model"] not in models:
            models.append(ev["model"])
    title = (conv.get("title") or "").strip() or common.short_title(_first_user_text(events)) or "(untitled conversation)"
    meta = {}
    model = conv.get("default_model_slug") or (models[0] if models else "")
    if model:
        meta["model"] = model
    meta["source"] = "ChatGPT export"
    if conv.get("gizmo_id"):
        meta["gpt"] = conv["gizmo_id"]
    session = {
        "agent": "chatgpt",
        "id": conv_id,
        "file": SESSION_SCHEME + conv_id,
        "title": title,
        "meta": meta,
        "events": events,
        "models": models,
        "n_records": sum(1 for n in (conv.get("mapping") or {}).values() if n.get("message")),
        "export_index": export_index,
        "create_time": conv.get("create_time"),
        "update_time": conv.get("update_time") or conv.get("create_time"),
        "is_archived": bool(conv.get("is_archived")),
    }
    return session


def _summary(session: dict) -> dict:
    events = session["events"]
    top = [ev for ev in events if ev.get("kind") != "branch"]
    tools = [b for ev in top for b in ev.get("blocks") or [] if b.get("type") == "tool_use"]
    timestamps = [ev["ts"] for ev in top if ev.get("ts")]
    first_ts = _iso(session.get("create_time")) or (timestamps[0] if timestamps else None)
    last_ts = (timestamps[-1] if timestamps else None) or _iso(session.get("update_time"))
    summary = common.make_summary(
        agent="chatgpt",
        id=session["id"],
        file=session["file"],
        title=session["title"],
        version="",
        first_ts=first_ts,
        last_ts=last_ts,
        n_user=sum(1 for ev in top if ev.get("kind") == "user"),
        n_assistant=sum(1 for ev in top if ev.get("kind") == "assistant"),
        n_tool=len(tools),
        n_web=sum(1 for b in tools if _WEB_TOOL_RE.search(str(b.get("name") or ""))),
        n_records=session["n_records"],
        model=session["meta"].get("model", ""),
        mtime=session.get("update_time") or 0,
    )
    if session.get("is_archived"):
        summary["archived"] = True
    return summary


# ---------------------------------------------------------------------------
# Public API (server.py)
# ---------------------------------------------------------------------------
def _load() -> tuple[list[Export], dict, list]:
    """All sessions from every configured export, reparsed only on change."""
    global _STATE
    exports = discover_exports()
    identity = tuple(e.identity() for e in exports)
    with _LOCK:
        if _STATE is not None and _STATE[0] == identity:
            return _STATE[1], _STATE[2], _STATE[3]
        sessions: dict[str, dict] = {}
        for index, export in enumerate(exports):
            for conv in load_conversations(export):
                session = build_session(conv, index)
                if session is None:
                    continue
                # The same conversation in two exports: keep the newer copy.
                prior = sessions.get(session["id"])
                if prior is None or (session["update_time"] or 0) >= (prior["update_time"] or 0):
                    sessions[session["id"]] = session
        summaries = [_summary(s) for s in sessions.values()]
        _STATE = (identity, exports, sessions, summaries)
        return exports, sessions, summaries


def list_sessions() -> list[dict]:
    """Summaries for every conversation in every configured export, unsorted."""
    return [dict(s) for s in _load()[2]]


def _sniff_image_mime(raw: bytes) -> str:
    if raw.startswith(b"\x89PNG"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return ""


def _hydrated_block(block: dict, export: Export | None, index: dict) -> dict:
    out = {k: v for k, v in block.items() if k != "asset"}
    member = index.get(block.get("asset") or "")
    reason = "not included in the export"
    if export is not None and member:
        try:
            raw = export.read(member)
        except (OSError, KeyError, zipfile.BadZipFile):
            raw = b""
        mime = _sniff_image_mime(raw) or (block.get("mime_type") or "")
        if raw and len(raw) > MAX_IMAGE_BYTES:
            reason = "too large to show"
        elif raw and mime.startswith("image/"):
            out["data_uri"] = f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")
            return out
    label = block.get("name") or block.get("asset") or "image"
    if export is None:
        return {"type": "text", "text": f"*[image: {label}]*"}
    return {"type": "text", "text": f"*[image: {label} — {reason}]*"}


def _hydrate(events: list[dict], export: Export | None, index: dict) -> list[dict]:
    out = []
    for ev in events:
        if ev.get("kind") == "branch":
            ev = dict(ev, groups=[_hydrate(g, export, index) for g in ev.get("groups") or []])
        elif any(b.get("type") == "image" for b in ev.get("blocks") or []):
            ev = dict(ev, blocks=[
                _hydrated_block(b, export, index) if b.get("type") == "image" else b
                for b in ev["blocks"]
            ])
        out.append(ev)
    return out


def parse_session_by_id(conversation_id: str, *, with_images: bool = True) -> dict | None:
    """Full parse of one conversation addressed as ``chatgpt:<id>``.

    ``with_images=False`` skips reading image files out of the export (search
    and Markdown export don't need them); image blocks then become text notes.
    """
    exports, sessions, _ = _load()
    session = sessions.get(conversation_id)
    if session is None:
        return None
    export = exports[session["export_index"]] if session["export_index"] < len(exports) else None
    has_images = any(b.get("type") == "image" for _, b in _walk_blocks(session["events"]))
    index = export.asset_index() if (with_images and has_images and export) else {}
    out = {k: v for k, v in session.items() if k not in ("export_index", "create_time", "update_time")}
    if has_images:
        out["events"] = _hydrate(session["events"], export if with_images else None, index)
    return out
