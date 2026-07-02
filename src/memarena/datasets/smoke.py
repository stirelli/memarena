from __future__ import annotations

import hashlib
import json
import random

from memarena.datasets.base import DatasetLoader, QAItem, Session

# Self-authored, 20-item synthetic smoke set (§5.2, §8 Day 1) — no license
# friction, no download, ships with the package. Stratified 4 items per
# question type matching LongMemEval's own strata (§5.6): single-session,
# multi-session, knowledge-update, temporal-reasoning, abstention.
_RAW_ITEMS: list[dict] = [
    # --- single_session ---
    {
        "id": "smoke-001", "namespace": "smoke-user-01", "question_type": "single_session",
        "sessions": [{
            "session_id": "s1", "timestamp": "2026-01-05T10:00:00Z",
            "messages": [
                {"role": "user", "content": "My dog's name is Biscuit and he's a golden retriever."},
                {"role": "assistant", "content": "Biscuit sounds lovely! Golden retrievers are wonderful family dogs."},
                {"role": "user", "content": "By the way, I'm allergic to peanuts."},
                {"role": "assistant", "content": "Noted, I'll keep that in mind for any food recommendations."},
            ],
        }],
        "question": "What is my dog's name?",
        "gold_evidence": ["My dog's name is Biscuit and he's a golden retriever."],
        "gold_answer": "Biscuit",
    },
    {
        "id": "smoke-002", "namespace": "smoke-user-02", "question_type": "single_session",
        "sessions": [{
            "session_id": "s1", "timestamp": "2026-01-06T10:00:00Z",
            "messages": [
                {"role": "user", "content": "My favorite programming language is Rust, I love the borrow checker."},
                {"role": "assistant", "content": "Rust's ownership model does take some getting used to."},
                {"role": "user", "content": "I also live in Lisbon these days."},
                {"role": "assistant", "content": "Lisbon is a great city for remote developers."},
            ],
        }],
        "question": "What is my favorite programming language?",
        "gold_evidence": ["My favorite programming language is Rust, I love the borrow checker."],
        "gold_answer": "Rust",
    },
    {
        "id": "smoke-003", "namespace": "smoke-user-03", "question_type": "single_session",
        "sessions": [{
            "session_id": "s1", "timestamp": "2026-01-07T10:00:00Z",
            "messages": [
                {"role": "user", "content": "I work as a backend engineer at a fintech startup."},
                {"role": "assistant", "content": "That's a demanding field, lots of correctness constraints."},
                {"role": "user", "content": "I also really enjoy hiking on weekends."},
                {"role": "assistant", "content": "Hiking is a great way to disconnect."},
            ],
        }],
        "question": "What do I do for work?",
        "gold_evidence": ["I work as a backend engineer at a fintech startup."],
        "gold_answer": "Backend engineer at a fintech startup",
    },
    {
        "id": "smoke-004", "namespace": "smoke-user-04", "question_type": "single_session",
        "sessions": [{
            "session_id": "s1", "timestamp": "2026-01-08T10:00:00Z",
            "messages": [
                {"role": "user", "content": "My wedding anniversary is March 3rd."},
                {"role": "assistant", "content": "I'll remember that for you."},
                {"role": "user", "content": "I prefer tea over coffee in the mornings."},
                {"role": "assistant", "content": "Noted, tea it is."},
            ],
        }],
        "question": "When is my wedding anniversary?",
        "gold_evidence": ["My wedding anniversary is March 3rd."],
        "gold_answer": "March 3rd",
    },
    # --- multi_session ---
    {
        "id": "smoke-005", "namespace": "smoke-user-05", "question_type": "multi_session",
        "sessions": [
            {
                "session_id": "s1", "timestamp": "2026-01-01T10:00:00Z",
                "messages": [{"role": "user", "content": "I just adopted a cat named Luna."},
                             {"role": "assistant", "content": "Congratulations on the new cat!"}],
            },
            {
                "session_id": "s2", "timestamp": "2026-01-20T10:00:00Z",
                "messages": [{"role": "user", "content": "I'm planning a trip to Japan next spring."},
                             {"role": "assistant", "content": "Spring is a beautiful time to visit Japan."}],
            },
        ],
        "question": "What is my cat's name?",
        "gold_evidence": ["I just adopted a cat named Luna."],
        "gold_answer": "Luna",
    },
    {
        "id": "smoke-006", "namespace": "smoke-user-06", "question_type": "multi_session",
        "sessions": [
            {
                "session_id": "s1", "timestamp": "2026-01-02T10:00:00Z",
                "messages": [{"role": "user", "content": "I'm training for a marathon in Berlin."},
                             {"role": "assistant", "content": "Good luck with the training block."}],
            },
            {
                "session_id": "s2", "timestamp": "2026-01-25T10:00:00Z",
                "messages": [{"role": "user", "content": "I just bought a new laptop for work."},
                             {"role": "assistant", "content": "Nice, what specs did you go for?"}],
            },
        ],
        "question": "What race am I training for?",
        "gold_evidence": ["I'm training for a marathon in Berlin."],
        "gold_answer": "A marathon in Berlin",
    },
    {
        "id": "smoke-007", "namespace": "smoke-user-07", "question_type": "multi_session",
        "sessions": [
            {
                "session_id": "s1", "timestamp": "2026-01-03T10:00:00Z",
                "messages": [{"role": "user", "content": "My sister's name is Elena and she's a nurse."},
                             {"role": "assistant", "content": "That's a demanding but rewarding job."}],
            },
            {
                "session_id": "s2", "timestamp": "2026-01-28T10:00:00Z",
                "messages": [{"role": "user", "content": "I switched to a standing desk last week."},
                             {"role": "assistant", "content": "How's your back feeling since the switch?"}],
            },
        ],
        "question": "What is my sister's profession?",
        "gold_evidence": ["My sister's name is Elena and she's a nurse."],
        "gold_answer": "Nurse",
    },
    {
        "id": "smoke-008", "namespace": "smoke-user-08", "question_type": "multi_session",
        "sessions": [
            {
                "session_id": "s1", "timestamp": "2026-01-04T10:00:00Z",
                "messages": [{"role": "user", "content": "I'm learning to play the violin."},
                             {"role": "assistant", "content": "The violin has a steep learning curve, keep at it."}],
            },
            {
                "session_id": "s2", "timestamp": "2026-01-29T10:00:00Z",
                "messages": [{"role": "user", "content": "I need to buy groceries for the week."},
                             {"role": "assistant", "content": "Meal prepping can save you some trips."}],
            },
        ],
        "question": "What instrument am I learning?",
        "gold_evidence": ["I'm learning to play the violin."],
        "gold_answer": "Violin",
    },
    # --- knowledge_update ---
    {
        "id": "smoke-009", "namespace": "smoke-user-09", "question_type": "knowledge_update",
        "sessions": [
            {
                "session_id": "s1", "timestamp": "2026-01-01T10:00:00Z",
                "messages": [{"role": "user", "content": "I live in Berlin."},
                             {"role": "assistant", "content": "Berlin has a great tech scene."}],
            },
            {
                "session_id": "s2", "timestamp": "2026-03-01T10:00:00Z",
                "messages": [{"role": "user", "content": "I just moved to Amsterdam."},
                             {"role": "assistant", "content": "Hope the move went smoothly!"}],
            },
        ],
        "question": "Where do I currently live?",
        "gold_evidence": ["I just moved to Amsterdam."],
        "gold_answer": "Amsterdam",
    },
    {
        "id": "smoke-010", "namespace": "smoke-user-10", "question_type": "knowledge_update",
        "sessions": [
            {
                "session_id": "s1", "timestamp": "2026-01-02T10:00:00Z",
                "messages": [{"role": "user", "content": "My phone number is 555-0142."},
                             {"role": "assistant", "content": "Got it, saved."}],
            },
            {
                "session_id": "s2", "timestamp": "2026-03-02T10:00:00Z",
                "messages": [{"role": "user", "content": "My phone number changed to 555-0199."},
                             {"role": "assistant", "content": "Updated, thanks for letting me know."}],
            },
        ],
        "question": "What is my current phone number?",
        "gold_evidence": ["My phone number changed to 555-0199."],
        "gold_answer": "555-0199",
    },
    {
        "id": "smoke-011", "namespace": "smoke-user-11", "question_type": "knowledge_update",
        "sessions": [
            {
                "session_id": "s1", "timestamp": "2026-01-03T10:00:00Z",
                "messages": [{"role": "user", "content": "I work at Acme Corp as a designer."},
                             {"role": "assistant", "content": "How do you like it there?"}],
            },
            {
                "session_id": "s2", "timestamp": "2026-03-03T10:00:00Z",
                "messages": [{"role": "user", "content": "I left Acme Corp and now work at Globex as a lead designer."},
                             {"role": "assistant", "content": "Congratulations on the new role!"}],
            },
        ],
        "question": "Where do I currently work?",
        "gold_evidence": ["I left Acme Corp and now work at Globex as a lead designer."],
        "gold_answer": "Globex",
    },
    {
        "id": "smoke-012", "namespace": "smoke-user-12", "question_type": "knowledge_update",
        "sessions": [
            {
                "session_id": "s1", "timestamp": "2026-01-04T10:00:00Z",
                "messages": [{"role": "user", "content": "My favorite restaurant is Trattoria Bella."},
                             {"role": "assistant", "content": "Great choice, their pasta is well known."}],
            },
            {
                "session_id": "s2", "timestamp": "2026-03-04T10:00:00Z",
                "messages": [
                    {"role": "user", "content": "Trattoria Bella closed down, my new favorite is Ramen House."},
                    {"role": "assistant", "content": "Sorry to hear that, Ramen House is a solid pick though."},
                ],
            },
        ],
        "question": "What is my favorite restaurant now?",
        "gold_evidence": ["Trattoria Bella closed down, my new favorite is Ramen House."],
        "gold_answer": "Ramen House",
    },
    # --- temporal_reasoning ---
    {
        "id": "smoke-013", "namespace": "smoke-user-13", "question_type": "temporal_reasoning",
        "sessions": [
            {
                "session_id": "s1", "timestamp": "2026-01-10T10:00:00Z",
                "messages": [{"role": "user", "content": "I started my new job on January 10th."},
                             {"role": "assistant", "content": "Congratulations on the new job!"}],
            },
            {
                "session_id": "s2", "timestamp": "2026-02-15T10:00:00Z",
                "messages": [{"role": "user", "content": "I got promoted after five weeks."},
                             {"role": "assistant", "content": "That's a fast promotion, well done."}],
            },
        ],
        "question": "When did I start my new job?",
        "gold_evidence": ["I started my new job on January 10th."],
        "gold_answer": "January 10th",
    },
    {
        "id": "smoke-014", "namespace": "smoke-user-14", "question_type": "temporal_reasoning",
        "sessions": [{
            "session_id": "s1", "timestamp": "2026-01-11T10:00:00Z",
            "messages": [
                {"role": "user", "content": "I adopted my dog three years ago, in 2023."},
                {"role": "assistant", "content": "Three years is a long, happy bond."},
            ],
        }],
        "question": "In what year did I adopt my dog?",
        "gold_evidence": ["I adopted my dog three years ago, in 2023."],
        "gold_answer": "2023",
    },
    {
        "id": "smoke-015", "namespace": "smoke-user-15", "question_type": "temporal_reasoning",
        "sessions": [{
            "session_id": "s1", "timestamp": "2026-01-12T10:00:00Z",
            "messages": [
                {"role": "user", "content": "My flight to Rome departs next Tuesday at 6am."},
                {"role": "assistant", "content": "I hope you get some rest before that early flight."},
            ],
        }],
        "question": "What time does my flight to Rome depart?",
        "gold_evidence": ["My flight to Rome departs next Tuesday at 6am."],
        "gold_answer": "6am",
    },
    {
        "id": "smoke-016", "namespace": "smoke-user-16", "question_type": "temporal_reasoning",
        "sessions": [{
            "session_id": "s1", "timestamp": "2026-01-13T10:00:00Z",
            "messages": [
                {"role": "user", "content": (
                    "I graduated from university in 2019, four years before starting my current job."
                )},
                {"role": "assistant", "content": "Sounds like a solid path to where you are now."},
            ],
        }],
        "question": "What year did I graduate from university?",
        "gold_evidence": ["I graduated from university in 2019, four years before starting my current job."],
        "gold_answer": "2019",
    },
    # --- abstention (never mentioned; gold requires abstaining) ---
    {
        "id": "smoke-017", "namespace": "smoke-user-17", "question_type": "abstention",
        "answerable": False,
        "sessions": [{
            "session_id": "s1", "timestamp": "2026-01-14T10:00:00Z",
            "messages": [
                {"role": "user", "content": "My dog's name is Rex."},
                {"role": "assistant", "content": "Rex is a great name for a dog."},
            ],
        }],
        "question": "What is my mother's maiden name?",
        "gold_evidence": [],
        "gold_answer": None,
    },
    {
        "id": "smoke-018", "namespace": "smoke-user-18", "question_type": "abstention",
        "answerable": False,
        "sessions": [{
            "session_id": "s1", "timestamp": "2026-01-15T10:00:00Z",
            "messages": [
                {"role": "user", "content": "I enjoy painting landscapes on weekends."},
                {"role": "assistant", "content": "Landscape painting is very meditative."},
            ],
        }],
        "question": "What car do I drive?",
        "gold_evidence": [],
        "gold_answer": None,
    },
    {
        "id": "smoke-019", "namespace": "smoke-user-19", "question_type": "abstention",
        "answerable": False,
        "sessions": [{
            "session_id": "s1", "timestamp": "2026-01-16T10:00:00Z",
            "messages": [
                {"role": "user", "content": "I'm reading a novel about deep-sea exploration."},
                {"role": "assistant", "content": "That sounds like an atmospheric read."},
            ],
        }],
        "question": "What is my blood type?",
        "gold_evidence": [],
        "gold_answer": None,
    },
    {
        "id": "smoke-020", "namespace": "smoke-user-20", "question_type": "abstention",
        "answerable": False,
        "sessions": [{
            "session_id": "s1", "timestamp": "2026-01-17T10:00:00Z",
            "messages": [
                {"role": "user", "content": "I collect vintage postcards from the 1960s."},
                {"role": "assistant", "content": "That's a charming hobby."},
            ],
        }],
        "question": "How many siblings do I have?",
        "gold_evidence": [],
        "gold_answer": None,
    },
]


class SmokeDatasetLoader(DatasetLoader):
    """Self-authored 20-item synthetic smoke set. No download, no license
    friction — lets CI and new users run instantly (§5.6)."""

    name = "smoke"
    origin_url = ""  # self-authored, nothing to fetch from origin
    revision = "v1"
    license = "Apache-2.0 (self-authored, part of this repository)"
    redistributable = True

    def sha256(self) -> str:
        canonical = json.dumps(_RAW_ITEMS, sort_keys=True).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def load(self, *, sample: int | None = None, seed: int = 42,
             stratify_by: str | None = "question_type") -> list[QAItem]:
        items = [self._to_qaitem(raw) for raw in _RAW_ITEMS]
        if sample is None or sample >= len(items):
            return items
        return self._stratified_sample(items, sample, seed, stratify_by)

    @staticmethod
    def _to_qaitem(raw: dict) -> QAItem:
        sessions = [
            Session(session_id=s["session_id"], timestamp=s["timestamp"], messages=s["messages"])
            for s in raw["sessions"]
        ]
        return QAItem(
            id=raw["id"],
            namespace=raw["namespace"],
            sessions=sessions,
            question=raw["question"],
            gold_evidence=raw["gold_evidence"],
            question_type=raw["question_type"],
            answerable=raw.get("answerable", True),
            gold_answer=raw.get("gold_answer"),
        )

    @staticmethod
    def _stratified_sample(items: list[QAItem], sample: int, seed: int,
                            stratify_by: str | None) -> list[QAItem]:
        rng = random.Random(seed)
        if not stratify_by:
            return sorted(rng.sample(items, sample), key=lambda i: i.id)

        strata: dict[str, list[QAItem]] = {}
        for item in items:
            strata.setdefault(getattr(item, stratify_by), []).append(item)

        n_strata = len(strata)
        base_quota = sample // n_strata
        remainder = sample % n_strata
        selected: list[QAItem] = []
        for i, (_, bucket) in enumerate(sorted(strata.items())):
            quota = base_quota + (1 if i < remainder else 0)
            quota = min(quota, len(bucket))
            selected.extend(rng.sample(bucket, quota))
        return sorted(selected, key=lambda i: i.id)
