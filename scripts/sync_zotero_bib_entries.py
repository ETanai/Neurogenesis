#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import socket
from pathlib import Path
from typing import Any


ENTRY_RE = re.compile(r"@(?P<type>[A-Za-z]+)\s*\{\s*(?P<key>[^,\s]+)\s*,(?P<body>.*?)\n\}", re.S)
FIELD_RE = re.compile(
    r"(?im)^\s*(?P<name>[A-Za-z][A-Za-z0-9_-]*)\s*=\s*"
    r"(?P<value>\{(?:[^{}]|(?:\{[^{}]*\}))*\}|\"[^\"]*\"|[^,\n]+)\s*,?\s*$"
)


def normalize_title(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.replace("{", "").replace("}", "").lower()).split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync missing BibTeX entries into a local Zotero group and verify the exported .bib file."
    )
    parser.add_argument("--group-name", required=True)
    parser.add_argument("--bib-file", required=True)
    parser.add_argument("--citekeys", nargs="+", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--import-bib")
    source.add_argument("--bibtex-literal")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--restart-zotero-on-session-exists", action="store_true")
    parser.add_argument("--zotero-base-url", default="http://127.0.0.1:23119")
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--diagnose", action="store_true", help="Print local Zotero HTTP-server diagnostics and exit.")
    return parser.parse_args()


def _clean_value(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("{") and raw.endswith("}"):
        raw = raw[1:-1]
    elif raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]
    return " ".join(raw.replace("\n", " ").split())


def parse_bibtex_entries(text: str) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    for match in ENTRY_RE.finditer(text):
        fields: dict[str, str] = {
            "entrytype": match.group("type"),
            "citekey": match.group("key"),
            "raw": match.group(0).strip(),
        }
        for field_match in FIELD_RE.finditer(match.group("body")):
            fields[field_match.group("name").lower()] = _clean_value(field_match.group("value"))
        entries[fields["citekey"]] = fields
    return entries


def http_json(url: str, *, params: dict[str, Any] | None = None, timeout: int = 20) -> Any:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "neurogenesis-zotero-sync/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def http_post_text(url: str, text: str, *, timeout: int = 180) -> tuple[int, str]:
    body = text.encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "text/plain; charset=utf-8",
            "User-Agent": "neurogenesis-zotero-sync/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def ensure_zotero_online(base_url: str) -> None:
    deadline = time.time() + 15
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            http_json(f"{base_url}/api/users/0/groups", timeout=5)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1)
    diagnostic = diagnose_local_zotero(base_url)
    raise RuntimeError(
        f"Zotero local API not reachable at {base_url}: {last_error}\n\n{diagnostic}"
    )


def _zotero_prefs_paths() -> list[Path]:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return []
    profiles = Path(appdata) / "Zotero" / "Zotero" / "Profiles"
    if not profiles.exists():
        return []
    return list(profiles.glob("*/prefs.js"))


def _pref_value(text: str, key: str) -> str | None:
    match = re.search(rf'user_pref\("{re.escape(key)}",\s*([^)]+)\);', text)
    return match.group(1).strip() if match else None


def _profile_pref_value(profile_prefs: Path, key: str) -> tuple[str | None, str]:
    value: str | None = None
    source = profile_prefs.name
    for path in (profile_prefs, profile_prefs.with_name("user.js")):
        if not path.exists():
            continue
        try:
            candidate = _pref_value(path.read_text(encoding="utf-8", errors="replace"), key)
        except OSError:
            continue
        if candidate is not None:
            value = candidate
            source = path.name
    return value, source


def _port_accepts_connections(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def _zotero_process_hint() -> str:
    if os.name != "nt":
        return "unknown on this platform"
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq zotero.exe"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "unknown"
    return "running" if "zotero.exe" in result.stdout.lower() else "not detected"


def diagnose_local_zotero(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    expected_port = str(parsed.port or 23119)
    expected_host = parsed.hostname or "127.0.0.1"
    lines = [
        "Local Zotero diagnostics:",
        f"- Expected local API URL: {base_url}",
        f"- Zotero process: {_zotero_process_hint()}",
        f"- Port {expected_host}:{expected_port} accepting connections: "
        f"{_port_accepts_connections(expected_host, int(expected_port))}",
    ]
    prefs_paths = _zotero_prefs_paths()
    if not prefs_paths:
        lines.append("- Zotero profile prefs.js was not found under APPDATA.")
    for prefs in prefs_paths:
        server_enabled, server_enabled_source = _profile_pref_value(
            prefs, "extensions.zotero.httpServer.enabled"
        )
        server_port, server_port_source = _profile_pref_value(prefs, "extensions.zotero.httpServer.port")
        local_api_enabled, local_api_source = _profile_pref_value(
            prefs, "extensions.zotero.httpServer.localAPI.enabled"
        )
        lines.extend(
            [
                f"- Profile: {prefs}",
                f"  extensions.zotero.httpServer.enabled = {server_enabled or '<missing>'} ({server_enabled_source})",
                f"  extensions.zotero.httpServer.port = {server_port or '<missing>'} ({server_port_source})",
                f"  extensions.zotero.httpServer.localAPI.enabled = {local_api_enabled or '<missing>'} ({local_api_source})",
            ]
        )
        if server_enabled != "true" or server_port != expected_port:
            lines.extend(
                [
                    "- Zotero 7 needs the connector HTTP server enabled, not only the local API flag.",
                    "- In Zotero, open Settings > Advanced > Config Editor and set:",
                    "  extensions.zotero.httpServer.enabled = true",
                    f"  extensions.zotero.httpServer.port = {expected_port}",
                    "  extensions.zotero.httpServer.localAPI.enabled = true",
                    "- Then restart Zotero and rerun this script.",
                ]
            )
    return "\n".join(lines)


def find_group(base_url: str, group_name: str) -> dict[str, Any]:
    groups = http_json(f"{base_url}/api/users/0/groups", timeout=10)
    for group in groups:
        data = group.get("data", {})
        if data.get("name") == group_name:
            return data
    names = [group.get("data", {}).get("name") for group in groups]
    raise RuntimeError(f"Zotero group '{group_name}' not found. Available groups: {names}")


def find_existing_in_group(base_url: str, group_id: int, entry: dict[str, str]) -> dict[str, Any] | None:
    queries: list[tuple[str, str]] = []
    citekey = entry.get("citekey", "").strip()
    doi = entry.get("doi", "").strip()
    title = entry.get("title", "").strip()
    if citekey:
        queries.append(("everything", citekey))
    if doi:
        queries.append(("everything", doi))
    if title:
        queries.append(("titleCreatorYear", title))
        queries.append(("everything", title))
    for mode, query in queries:
        items = http_json(
            f"{base_url}/api/groups/{group_id}/items",
            params={"q": query, "qmode": mode, "limit": 20},
            timeout=60,
        )
        for item in items:
            data = item.get("data", {})
            if citekey and (
                data.get("citationKey", "").strip() == citekey
                or f"Citation Key: {citekey}" in data.get("extra", "")
            ):
                return item
            if doi and data.get("DOI", "").strip().lower() == doi.lower():
                return item
            if title and normalize_title(data.get("title", "")) == normalize_title(title):
                return item
    return None


def start_select_group(group_id: int) -> None:
    if os.name == "nt":
        os.startfile(f"zotero://select/groups/{group_id}")  # type: ignore[attr-defined]
    else:
        subprocess.run(["xdg-open", f"zotero://select/groups/{group_id}"], check=False)


def restart_zotero_windows() -> None:
    subprocess.run(["taskkill", "/IM", "zotero.exe", "/F"], check=False, capture_output=True)
    time.sleep(2)
    subprocess.Popen([r"C:\Program Files\Zotero\zotero.exe"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(8)


def connector_import(base_url: str, bibtex: str, *, restart_on_session_exists: bool) -> list[dict[str, Any]]:
    status, text = http_post_text(f"{base_url}/connector/import", bibtex)
    if status == 201:
        return json.loads(text)
    if status == 409 and "SESSION_EXISTS" in text and restart_on_session_exists and os.name == "nt":
        restart_zotero_windows()
        ensure_zotero_online(base_url)
        status, text = http_post_text(f"{base_url}/connector/import", bibtex)
        if status == 201:
            return json.loads(text)
    raise RuntimeError(f"Zotero connector import failed: HTTP {status} {text[:500]}")


def wait_for_bib_entries(path: Path, citekeys: list[str], timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            text = path.read_text(encoding="utf-8-sig")
            if all(re.search(rf"@\w+\{{{re.escape(key)}\s*,", text) for key in citekeys):
                return True
        time.sleep(1)
    return False


def load_source(path: str | None) -> str:
    if path is None:
        raise ValueError("Missing BibTeX source path")
    return Path(path).read_text(encoding="utf-8-sig")


def main() -> int:
    args = parse_args()
    if args.diagnose:
        print(diagnose_local_zotero(args.zotero_base_url))
        return 0
    bib_path = Path(args.bib_file)
    bib_text = load_source(args.import_bib or args.bibtex_literal)
    entries = parse_bibtex_entries(bib_text)
    requested = sorted({key.strip() for key in args.citekeys})
    missing_from_source = [key for key in requested if key not in entries]
    if missing_from_source:
        raise SystemExit(f"Citekeys missing from BibTeX source: {missing_from_source}")

    ensure_zotero_online(args.zotero_base_url)
    group = find_group(args.zotero_base_url, args.group_name)

    existing: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, str]] = []
    for key in requested:
        entry = entries[key]
        hit = find_existing_in_group(args.zotero_base_url, int(group["id"]), entry)
        if hit:
            existing[key] = hit
        else:
            missing.append(entry)

    summary: dict[str, Any] = {
        "group_name": group["name"],
        "group_id": group["id"],
        "bib_file": str(bib_path),
        "requested_citekeys": requested,
        "existing_citekeys": sorted(existing),
        "missing_citekeys": [entry["citekey"] for entry in missing],
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    imported: list[dict[str, Any]] = []
    if missing:
        start_select_group(int(group["id"]))
        time.sleep(2)
        import_text = "\n\n".join(entry["raw"] for entry in missing)
        imported = connector_import(
            args.zotero_base_url,
            import_text,
            restart_on_session_exists=args.restart_zotero_on_session_exists,
        )
        time.sleep(4)

    if not wait_for_bib_entries(bib_path, requested, args.poll_seconds):
        raise SystemExit(
            f"Requested citekeys were not all present in {bib_path} after {args.poll_seconds}s. "
            "Refresh or export the Better BibTeX bibliography and rerun verification."
        )

    summary.update({"imported_item_keys": [item.get("key") for item in imported], "verified_in_bib": requested})
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
