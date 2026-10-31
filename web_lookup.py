#!/usr/bin/env python3
"""One shape for web searches and page fetches, whichever agent made them.

Every source records web activity differently: Claude Code's WebSearch puts a
``Links: [...]`` JSON list and a written summary in its result text and its
WebFetch answers a prompt about a URL; Codex logs ``web_search_call`` actions
(search / open_page / find_in_page) without their results; Cursor and opencode
have their own tool names and argument spellings; ChatGPT exports keep the
retrieved pages but not the query. ``annotate()`` adds a ``web`` dict to each
such tool block or event so the frontend and the Markdown export render all of
them one way:

``{"kind": "search"|"fetch", "queries": [str], "url": str, "prompt": str,
"pattern": str, "pages": [{"title", "url", "snippet"}], "sites": [str],
"text": str, "text_label": str, "recorded": bool, "note": str}``

``pages`` are the results a search returned (or the page a fetch read);
``text`` is the rest of the result — the agent's summary of a search, the
content or answer of a fetch. ``recorded`` is False when the source never
writes results down (Codex), as opposed to a lookup that found nothing.
Fields that don't apply are omitted.
"""

from __future__ import annotations

import json
import re

_SEARCH_TOOLS = {"websearch", "web_search", "search_web", "web.search", "google_search"}
_FETCH_TOOLS = {"webfetch", "web_fetch", "fetch", "fetch_url", "web.open", "open_url", "read_url"}

_QUERY_KEYS = ("query", "search_term", "searchTerm", "q")
_URL_KEYS = ("url", "uri", "href")

_LINKS_LINE_RE = re.compile(r"^Links:\s*(\[.*\])\s*$", re.M)
_CLAUDE_HEADER_RE = re.compile(r'^Web search results for query: ".*"\s*$', re.M)
# Exa-style results (opencode's websearch): "Title: …\nURL: …\n…Text: …".
_TITLE_URL_RE = re.compile(r"^Title:\s*(.+?)\s*\n(?:.*\n){0,3}?URL:\s*(\S+)", re.M)
_URL_RE = re.compile(r"https?://[^\s<>()\"'\]\[]+")


def _http(url) -> str:
    return url if isinstance(url, str) and url.startswith(("http://", "https://")) else ""


def page(title="", url="", snippet="") -> dict | None:
    url = _http(url)
    if not url:
        return None
    out = {"title": " ".join(str(title or "").split()) or url, "url": url}
    snippet = " ".join(str(snippet or "").split())
    if snippet:
        out["snippet"] = snippet[:300] + ("…" if len(snippet) > 300 else "")
    return out


def _dedupe(pages) -> list[dict]:
    out, seen = [], set()
    for p in pages:
        if p and p["url"] not in seen:
            seen.add(p["url"])
            out.append(p)
    return out


def _pages_from_json(value) -> list[dict]:
    """Every dict with an http(s) `url` anywhere in a decoded JSON value."""
    found = []
    if isinstance(value, dict):
        p = page(value.get("title") or value.get("name"), value.get("url") or value.get("link"),
                 value.get("snippet") or value.get("chunk") or value.get("description") or value.get("text"))
        if p:
            found.append(p)
        for v in value.values():
            if isinstance(v, (dict, list)):
                found.extend(_pages_from_json(v))
    elif isinstance(value, list):
        for v in value:
            found.extend(_pages_from_json(v))
    return found


def _title_url_pages(text: str) -> list[dict]:
    """Pages from Exa-style "Title: … / URL: … / Text: …" records, the text
    (or highlights) becoming the snippet, so nothing needs repeating raw."""
    pages = []
    for chunk in re.split(r"(?m)^(?=Title:)", text):
        fields: dict[str, list[str]] = {}
        key = None
        for line in chunk.splitlines():
            m = re.match(r"^(Title|URL|Text|Highlights|Summary|Published Date|Author):\s*(.*)$", line)
            if m:
                key = m.group(1)
                fields.setdefault(key, []).append(m.group(2))
            elif key in ("Text", "Highlights", "Summary"):
                fields[key].append(line)
        url = (fields.get("URL") or [""])[0]
        snippet = " ".join(fields.get("Summary") or fields.get("Highlights") or fields.get("Text") or [])
        pages.append(page((fields.get("Title") or [""])[0], url, snippet))
    return pages


def search_results(text: str) -> tuple[list[dict], str]:
    """(pages, remaining text) from a search tool's result text."""
    if not isinstance(text, str) or not text.strip():
        return [], ""
    pages: list[dict] = []
    rest = text
    # Claude Code: one "Links: [...]" line per search the tool ran.
    for m in _LINKS_LINE_RE.finditer(text):
        try:
            pages.extend(_pages_from_json(json.loads(m.group(1))))
        except ValueError:
            continue
    if pages:
        rest = _LINKS_LINE_RE.sub("", _CLAUDE_HEADER_RE.sub("", text))
    else:
        stripped = text.strip()
        if stripped[:1] in "[{":
            try:
                pages = _pages_from_json(json.loads(stripped))
                rest = ""
            except ValueError:
                pages = []
        if not pages and _TITLE_URL_RE.search(text):
            pages, rest = _title_url_pages(text), ""
        if not any(pages):
            pages = [page("", u.rstrip(".,;:")) for u in _URL_RE.findall(text)]
    rest = re.sub(r"\n{3,}", "\n\n", rest).strip()
    return _dedupe(pages), rest


def _first(value: dict, keys) -> str:
    for key in keys:
        v = value.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def tool_kind(name) -> str:
    n = str(name or "").strip().lower()
    if n in _SEARCH_TOOLS:
        return "search"
    if n in _FETCH_TOOLS:
        return "fetch"
    return ""


def from_tool(name, value, result) -> dict | None:
    """The `web` dict for a block-shaped tool call, or None if it isn't one."""
    kind = tool_kind(name)
    if not kind:
        return None
    value = value if isinstance(value, dict) else {}
    text = ""
    if isinstance(result, dict):
        text = result.get("text") or result.get("output") or ""
        if not isinstance(text, str):
            text = json.dumps(text, indent=2)
    web: dict = {"kind": kind, "recorded": result is not None}
    if kind == "search":
        queries = value.get("queries") if isinstance(value.get("queries"), list) else []
        queries = [q for q in queries if isinstance(q, str) and q.strip()] or [q for q in [_first(value, _QUERY_KEYS)] if q]
        web["queries"] = queries
        pages, rest = search_results(text)
        web["pages"] = pages
        if rest:
            # Only Claude Code writes a summary after its links; elsewhere the
            # leftover text is just the tool's raw output.
            web["text"], web["text_label"] = rest, ("Summary" if _LINKS_LINE_RE.search(text) else "Output")
    else:
        url = _first(value, _URL_KEYS)
        web["url"] = url
        prompt = _first(value, ("prompt", "question"))
        if prompt:
            web["prompt"] = prompt
        p = page("", url)
        web["pages"] = [p] if p else []
        if text.strip():
            web["text"], web["text_label"] = text.strip(), ("Answer" if prompt else "Content")
    if isinstance(result, dict) and result.get("is_error"):
        web["error"] = True
    return web


def from_codex(ev: dict) -> dict | None:
    """The `web` dict for a Codex web_search / web_call event."""
    action = ev.get("action") if isinstance(ev.get("action"), dict) else {}
    atype = action.get("type") or "search"
    note = "Codex records the request, not the results it returned."
    if atype in ("open_page", "find_in_page"):
        url = _http(action.get("url"))
        web = {"kind": "fetch", "url": url, "pages": [page("", url)] if url else [], "recorded": False, "note": note}
        if action.get("pattern"):
            web["pattern"] = str(action["pattern"])
        return web
    queries = [q for q in action.get("queries") or [] if isinstance(q, str) and q.strip()]
    if not queries:
        queries = [q for q in (ev.get("query"), action.get("query")) if isinstance(q, str) and q.strip()][:1]
    return {"kind": "search", "queries": queries, "pages": [], "recorded": False, "note": note}


def annotate(data: dict) -> dict:
    """Add `web` to every web lookup in a parsed session, in place."""
    def walk(events):
        for ev in events or []:
            kind = ev.get("kind")
            if kind == "branch":
                for group in ev.get("groups") or []:
                    walk(group)
            elif kind in ("web_search", "web_call"):
                ev.setdefault("web", from_codex(ev))
            elif kind == "tool" and "web" not in ev:
                web = from_tool(ev.get("name"), ev.get("input"), ev.get("result"))
                if web:
                    ev["web"] = web
            for b in ev.get("blocks") or []:
                if b.get("type") == "tool_use" and "web" not in b:
                    web = from_tool(b.get("name"), b.get("input"), b.get("result"))
                    if web:
                        b["web"] = web

    walk(data.get("events"))
    return data
