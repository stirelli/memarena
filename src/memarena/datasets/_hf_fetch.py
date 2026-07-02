from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx

HF_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-v2/resolve/{revision}/{path}"
CHECKSUMS_PATH = "checksums.sha256"


class ChecksumMismatchError(Exception):
    pass


class HuggingFaceFetcher:
    """Real origin fetcher for LongMemEval-V2 (§5.6: download-from-origin,
    verify sha256, cache locally). Never redistributes — only caches under
    `cache_dir`, which is gitignored."""

    def __init__(self, *, revision: str, cache_dir: Path):
        self._revision = revision
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._checksums: dict[str, str] | None = None

    def _url(self, path: str) -> str:
        return HF_BASE.format(revision=self._revision, path=path)

    def _load_checksums(self) -> dict[str, str]:
        if self._checksums is None:
            resp = httpx.get(self._url(CHECKSUMS_PATH), timeout=30.0, follow_redirects=True)
            resp.raise_for_status()
            checksums = {}
            for line in resp.text.splitlines():
                if not line.strip():
                    continue
                digest, name = line.split(maxsplit=1)
                checksums[name.strip()] = digest.strip()
            self._checksums = checksums
        return self._checksums

    def _fetch_and_verify_full_file(self, remote_path: str, local_name: str) -> Path:
        local_path = self._cache_dir / local_name
        if local_path.exists():
            return local_path
        resp = httpx.get(self._url(remote_path), timeout=120.0, follow_redirects=True)
        resp.raise_for_status()
        digest = hashlib.sha256(resp.content).hexdigest()
        expected = self._load_checksums().get(remote_path)
        if expected and digest != expected:
            raise ChecksumMismatchError(f"{remote_path}: expected {expected}, got {digest}")
        local_path.write_bytes(resp.content)
        return local_path

    def fetch_questions(self) -> str:
        return str(self._fetch_and_verify_full_file("questions.jsonl", "questions.jsonl"))

    def fetch_haystack_small(self) -> str:
        return str(self._fetch_and_verify_full_file("haystacks/lme_v2_small.json", "haystack_small.json"))

    def fetch_trajectories(self, needed_ids: set[str]) -> str:
        cache_key = hashlib.sha256(",".join(sorted(needed_ids)).encode()).hexdigest()[:16]
        local_path = self._cache_dir / f"trajectories_filtered_{cache_key}.jsonl"
        if local_path.exists():
            return str(local_path)

        expected = self._load_checksums().get("trajectories.jsonl")
        hasher = hashlib.sha256()
        matched_lines: list[bytes] = []
        with httpx.stream("GET", self._url("trajectories.jsonl"), timeout=600.0, follow_redirects=True) as resp:
            resp.raise_for_status()
            buffer = b""
            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                hasher.update(chunk)
                buffer += chunk
                *complete_lines, buffer = buffer.split(b"\n")
                for raw_line in complete_lines:
                    if not raw_line.strip():
                        continue
                    obj = json.loads(raw_line)
                    if obj["id"] in needed_ids:
                        matched_lines.append(raw_line)
            if buffer.strip():
                obj = json.loads(buffer)
                if obj["id"] in needed_ids:
                    matched_lines.append(buffer)

        digest = hasher.hexdigest()
        if expected and digest != expected:
            raise ChecksumMismatchError(f"trajectories.jsonl: expected {expected}, got {digest}")

        with local_path.open("wb") as out:
            for line in matched_lines:
                out.write(line + b"\n")
        return str(local_path)
