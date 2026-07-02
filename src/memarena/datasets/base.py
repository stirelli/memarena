from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Session:
    session_id: str
    timestamp: str
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class QAItem:
    id: str
    namespace: str
    sessions: list[Session]
    question: str
    gold_evidence: list[str]
    question_type: str
    answerable: bool = True
    gold_answer: str | None = None


class DatasetLoader(ABC):
    """Contract for a benchmark dataset (§5.6).

    Ships zero dataset content for licensed/third-party datasets — only
    the loader. First run downloads from `origin_url` at a pinned
    revision, verifies against `sha256()`, and caches locally.
    Self-authored synthetic datasets (e.g. the smoke set) have no
    external origin and no license friction.
    """

    name: str = ""
    origin_url: str = ""
    revision: str = ""
    license: str = ""
    redistributable: bool = False

    @abstractmethod
    def sha256(self) -> str:
        """SHA-256 digest of the dataset artifact backing this loader."""

    @abstractmethod
    def load(self, *, sample: int | None = None, seed: int = 42,
             stratify_by: str | None = None) -> list[QAItem]:
        """Return the (optionally stratified-sampled) list of QA items."""


__all__ = ["DatasetLoader", "QAItem", "Session"]
