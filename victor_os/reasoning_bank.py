"""
Victor Reasoning Bank Organ
===========================

Offline-first procedural memory for VictorOS.

The organ stores:
1. immutable task experiences (including failures),
2. distilled, reusable reasoning rules,
3. provenance linking rules back to the experiences that support them,
4. supersession history for safe rule evolution.

Design goals:
- Standard-library only (sqlite3, hashlib, json).
- WAL-backed durable storage.
- Failures are first-class evidence.
- Retrieval is deterministic and explainable.
- New rules are quarantined until minimum evidence is met.
- Existing rules are never silently overwritten; revisions are auditable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple


Outcome = Literal["success", "failure", "partial", "unknown"]
RuleStatus = Literal["candidate", "active", "deprecated", "rejected"]

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{2,}")


def _now() -> float:
    return time.time()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: Any) -> str:
    raw = value if isinstance(value, str) else _canonical_json(value)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")}


@dataclass(frozen=True)
class Experience:
    task: str
    action: str
    outcome: Outcome
    observation: str
    context: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    episode_id: Optional[str] = None
    created_at: float = field(default_factory=_now)
    experience_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def validate(self) -> None:
        if not self.task.strip():
            raise ValueError("task must not be empty")
        if not self.action.strip():
            raise ValueError("action must not be empty")
        if self.outcome not in {"success", "failure", "partial", "unknown"}:
            raise ValueError(f"invalid outcome: {self.outcome}")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")


@dataclass(frozen=True)
class ReasoningRule:
    title: str
    trigger: str
    strategy: str
    rationale: str
    tags: Sequence[str] = field(default_factory=tuple)
    confidence: float = 0.5
    status: RuleStatus = "candidate"
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    success_count: int = 0
    failure_count: int = 0
    partial_count: int = 0
    unknown_count: int = 0
    supersedes: Optional[str] = None

    def validate(self) -> None:
        for name, value in (
            ("title", self.title),
            ("trigger", self.trigger),
            ("strategy", self.strategy),
            ("rationale", self.rationale),
        ):
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.status not in {"candidate", "active", "deprecated", "rejected"}:
            raise ValueError(f"invalid rule status: {self.status}")


@dataclass(frozen=True)
class RetrievedRule:
    rule: ReasoningRule
    relevance: float
    overlap_terms: Tuple[str, ...]
    evidence_count: int


class ReasoningBank:
    """Durable procedural-memory organ.

    A single instance is safe for multi-threaded use inside one process.
    SQLite WAL allows concurrent readers from other processes.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        db_path: str | Path = "memory/reasoning_bank.sqlite3",
        *,
        min_evidence_to_activate: int = 2,
        min_confidence_to_activate: float = 0.60,
    ) -> None:
        if min_evidence_to_activate < 1:
            raise ValueError("min_evidence_to_activate must be >= 1")
        if not 0 <= min_confidence_to_activate <= 1:
            raise ValueError("min_confidence_to_activate must be in [0, 1]")

        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.min_evidence_to_activate = min_evidence_to_activate
        self.min_confidence_to_activate = min_confidence_to_activate
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._configure()
        self._migrate()

    def _configure(self) -> None:
        with self._db:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.execute("PRAGMA busy_timeout=5000")

    def _migrate(self) -> None:
        with self._lock, self._db:
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS experiences (
                    experience_id TEXT PRIMARY KEY,
                    episode_id TEXT,
                    task TEXT NOT NULL,
                    action TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    observation TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    score REAL NOT NULL,
                    evidence_hash TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS rules (
                    rule_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    trigger_text TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    partial_count INTEGER NOT NULL DEFAULT 0,
                    unknown_count INTEGER NOT NULL DEFAULT 0,
                    supersedes TEXT,
                    FOREIGN KEY(supersedes) REFERENCES rules(rule_id)
                )
                """
            )
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS rule_evidence (
                    rule_id TEXT NOT NULL,
                    experience_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(rule_id, experience_id),
                    FOREIGN KEY(rule_id) REFERENCES rules(rule_id) ON DELETE CASCADE,
                    FOREIGN KEY(experience_id) REFERENCES experiences(experience_id)
                )
                """
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiences_episode ON experiences(episode_id)"
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_rules_status ON rules(status)"
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_rule_evidence_rule ON rule_evidence(rule_id)"
            )
            self._db.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(self.SCHEMA_VERSION),),
            )

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> "ReasoningBank":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def record_experience(self, experience: Experience) -> str:
        """Append an immutable experience and return its ID."""
        experience.validate()
        evidence_hash = _sha256(experience.evidence)
        with self._lock, self._db:
            self._db.execute(
                """
                INSERT INTO experiences(
                    experience_id, episode_id, task, action, outcome, observation,
                    context_json, evidence_json, score, evidence_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experience.experience_id,
                    experience.episode_id,
                    experience.task,
                    experience.action,
                    experience.outcome,
                    experience.observation,
                    _canonical_json(experience.context),
                    _canonical_json(experience.evidence),
                    float(experience.score),
                    evidence_hash,
                    float(experience.created_at),
                ),
            )
        return experience.experience_id

    def create_rule(
        self,
        rule: ReasoningRule,
        *,
        evidence_ids: Sequence[str] = (),
        evidence_relation: str = "supports",
    ) -> str:
        """Create a candidate/active rule and attach provenance."""
        rule.validate()
        evidence_ids = tuple(dict.fromkeys(evidence_ids))
        counts = self._outcome_counts(evidence_ids)
        evidence_count = sum(counts.values())
        confidence = self._posterior_confidence(counts, prior=rule.confidence)
        requested_status = rule.status
        status: RuleStatus = requested_status
        if requested_status == "active" and not self._activation_allowed(
            evidence_count, confidence
        ):
            status = "candidate"

        with self._lock, self._db:
            self._assert_experiences_exist(evidence_ids)
            self._db.execute(
                """
                INSERT INTO rules(
                    rule_id, title, trigger_text, strategy, rationale, tags_json,
                    confidence, status, created_at, updated_at,
                    success_count, failure_count, partial_count, unknown_count, supersedes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rule.rule_id,
                    rule.title,
                    rule.trigger,
                    rule.strategy,
                    rule.rationale,
                    _canonical_json(sorted(set(rule.tags))),
                    confidence,
                    status,
                    rule.created_at,
                    rule.updated_at,
                    counts["success"],
                    counts["failure"],
                    counts["partial"],
                    counts["unknown"],
                    rule.supersedes,
                ),
            )
            for experience_id in evidence_ids:
                self._db.execute(
                    """
                    INSERT INTO rule_evidence(rule_id, experience_id, relation, weight, created_at)
                    VALUES (?, ?, ?, 1.0, ?)
                    """,
                    (rule.rule_id, experience_id, evidence_relation, _now()),
                )
        return rule.rule_id

    def attach_evidence(
        self,
        rule_id: str,
        experience_id: str,
        *,
        relation: str = "supports",
        weight: float = 1.0,
    ) -> None:
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("weight must be a positive finite number")
        with self._lock, self._db:
            self._assert_rule_exists(rule_id)
            self._assert_experiences_exist((experience_id,))
            self._db.execute(
                """
                INSERT OR REPLACE INTO rule_evidence(
                    rule_id, experience_id, relation, weight, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (rule_id, experience_id, relation, float(weight), _now()),
            )
            self._recompute_rule(rule_id)

    def revise_rule(
        self,
        old_rule_id: str,
        *,
        title: Optional[str] = None,
        trigger: Optional[str] = None,
        strategy: Optional[str] = None,
        rationale: Optional[str] = None,
        tags: Optional[Sequence[str]] = None,
        status: RuleStatus = "candidate",
    ) -> str:
        """Create a new rule version and deprecate the old one.

        Evidence is copied forward, so provenance survives the revision.
        """
        old = self.get_rule(old_rule_id)
        evidence_ids = self.evidence_for_rule(old_rule_id)
        new_rule = ReasoningRule(
            title=title or old.title,
            trigger=trigger or old.trigger,
            strategy=strategy or old.strategy,
            rationale=rationale or old.rationale,
            tags=tuple(tags) if tags is not None else tuple(old.tags),
            confidence=old.confidence,
            status=status,
            supersedes=old_rule_id,
        )
        new_id = self.create_rule(new_rule, evidence_ids=evidence_ids)
        with self._lock, self._db:
            self._db.execute(
                "UPDATE rules SET status='deprecated', updated_at=? WHERE rule_id=?",
                (_now(), old_rule_id),
            )
        return new_id

    def promote_rule(self, rule_id: str) -> bool:
        """Promote a candidate only when evidence and confidence gates pass."""
        with self._lock, self._db:
            row = self._rule_row(rule_id)
            evidence_count = (
                row["success_count"]
                + row["failure_count"]
                + row["partial_count"]
                + row["unknown_count"]
            )
            allowed = self._activation_allowed(evidence_count, row["confidence"])
            if allowed:
                self._db.execute(
                    "UPDATE rules SET status='active', updated_at=? WHERE rule_id=?",
                    (_now(), rule_id),
                )
            return allowed

    def reject_rule(self, rule_id: str) -> None:
        with self._lock, self._db:
            self._assert_rule_exists(rule_id)
            self._db.execute(
                "UPDATE rules SET status='rejected', updated_at=? WHERE rule_id=?",
                (_now(), rule_id),
            )

    def retrieve(
        self,
        task: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        limit: int = 5,
        include_candidates: bool = False,
    ) -> List[RetrievedRule]:
        """Retrieve semantically-adjacent rules using explainable lexical scoring.

        This deliberately avoids hidden embeddings/dependencies. A caller may layer
        an embedding reranker on top later without changing storage semantics.
        """
        if limit < 1:
            raise ValueError("limit must be >= 1")
        query_blob = task + " " + _canonical_json(context or {})
        query_tokens = _tokens(query_blob)
        statuses = ("active", "candidate") if include_candidates else ("active",)
        placeholders = ",".join("?" for _ in statuses)

        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM rules WHERE status IN ({placeholders})",
                statuses,
            ).fetchall()

        results: List[RetrievedRule] = []
        for row in rows:
            tags = tuple(json.loads(row["tags_json"]))
            rule_blob = " ".join(
                [row["title"], row["trigger_text"], row["strategy"], " ".join(tags)]
            )
            rule_tokens = _tokens(rule_blob)
            overlap = query_tokens & rule_tokens
            union = query_tokens | rule_tokens
            lexical = len(overlap) / len(union) if union else 0.0
            trigger_bonus = 0.15 if _tokens(row["trigger_text"]) & query_tokens else 0.0
            confidence_component = 0.25 * float(row["confidence"])
            evidence_count = (
                row["success_count"]
                + row["failure_count"]
                + row["partial_count"]
                + row["unknown_count"]
            )
            evidence_component = 0.05 * min(1.0, math.log1p(evidence_count) / math.log(6))
            relevance = min(
                1.0, 0.55 * lexical + trigger_bonus + confidence_component + evidence_component
            )
            if overlap or relevance >= 0.25:
                results.append(
                    RetrievedRule(
                        rule=self._row_to_rule(row),
                        relevance=relevance,
                        overlap_terms=tuple(sorted(overlap)),
                        evidence_count=evidence_count,
                    )
                )
        results.sort(key=lambda r: (r.relevance, r.rule.confidence), reverse=True)
        return results[:limit]

    def get_rule(self, rule_id: str) -> ReasoningRule:
        with self._lock:
            return self._row_to_rule(self._rule_row(rule_id))

    def get_experience(self, experience_id: str) -> Experience:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM experiences WHERE experience_id=?",
                (experience_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown experience: {experience_id}")
        return Experience(
            task=row["task"],
            action=row["action"],
            outcome=row["outcome"],
            observation=row["observation"],
            context=json.loads(row["context_json"]),
            evidence=json.loads(row["evidence_json"]),
            score=float(row["score"]),
            episode_id=row["episode_id"],
            created_at=float(row["created_at"]),
            experience_id=row["experience_id"],
        )

    def evidence_for_rule(self, rule_id: str) -> List[str]:
        with self._lock:
            self._assert_rule_exists(rule_id)
            rows = self._db.execute(
                """
                SELECT experience_id FROM rule_evidence
                WHERE rule_id=? ORDER BY created_at ASC
                """,
                (rule_id,),
            ).fetchall()
        return [row["experience_id"] for row in rows]

    def verify_integrity(self) -> Dict[str, Any]:
        """Return a compact integrity receipt suitable for TRACE/Chronos logging."""
        with self._lock:
            pragma = self._db.execute("PRAGMA integrity_check").fetchone()[0]
            exp_count = self._db.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
            rule_count = self._db.execute("SELECT COUNT(*) FROM rules").fetchone()[0]
            active_count = self._db.execute(
                "SELECT COUNT(*) FROM rules WHERE status='active'"
            ).fetchone()[0]
            orphan_count = self._db.execute(
                """
                SELECT COUNT(*)
                FROM rule_evidence re
                LEFT JOIN rules r ON r.rule_id = re.rule_id
                LEFT JOIN experiences e ON e.experience_id = re.experience_id
                WHERE r.rule_id IS NULL OR e.experience_id IS NULL
                """
            ).fetchone()[0]
        receipt = {
            "ok": pragma == "ok" and orphan_count == 0,
            "sqlite_integrity": pragma,
            "experiences": exp_count,
            "rules": rule_count,
            "active_rules": active_count,
            "orphan_links": orphan_count,
            "schema_version": self.SCHEMA_VERSION,
        }
        receipt["receipt_hash"] = _sha256(receipt)
        return receipt

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            exp = dict(
                self._db.execute(
                    "SELECT outcome, COUNT(*) AS n FROM experiences GROUP BY outcome"
                ).fetchall()
            )
            rules = dict(
                self._db.execute(
                    "SELECT status, COUNT(*) AS n FROM rules GROUP BY status"
                ).fetchall()
            )
        return {"experiences": exp, "rules": rules}

    def _activation_allowed(self, evidence_count: int, confidence: float) -> bool:
        return (
            evidence_count >= self.min_evidence_to_activate
            and confidence >= self.min_confidence_to_activate
        )

    @staticmethod
    def _posterior_confidence(counts: Dict[str, int], *, prior: float = 0.5) -> float:
        weighted = (
            counts["success"]
            + 0.5 * counts["partial"]
            + 0.25 * counts["unknown"]
        )
        n = sum(counts.values())
        strength = 2.0
        value = (weighted + prior * strength) / (n + strength)
        return max(0.0, min(1.0, value))

    def _outcome_counts(self, experience_ids: Sequence[str]) -> Dict[str, int]:
        counts = {"success": 0, "failure": 0, "partial": 0, "unknown": 0}
        if not experience_ids:
            return counts
        with self._lock:
            placeholders = ",".join("?" for _ in experience_ids)
            rows = self._db.execute(
                f"""
                SELECT outcome, COUNT(*) AS n FROM experiences
                WHERE experience_id IN ({placeholders})
                GROUP BY outcome
                """,
                tuple(experience_ids),
            ).fetchall()
        for row in rows:
            counts[row["outcome"]] = int(row["n"])
        return counts

    def _assert_experiences_exist(self, experience_ids: Sequence[str]) -> None:
        if not experience_ids:
            return
        placeholders = ",".join("?" for _ in experience_ids)
        rows = self._db.execute(
            f"SELECT experience_id FROM experiences WHERE experience_id IN ({placeholders})",
            tuple(experience_ids),
        ).fetchall()
        found = {row["experience_id"] for row in rows}
        missing = [eid for eid in experience_ids if eid not in found]
        if missing:
            raise KeyError(f"unknown experience(s): {', '.join(missing)}")

    def _assert_rule_exists(self, rule_id: str) -> None:
        self._rule_row(rule_id)

    def _rule_row(self, rule_id: str) -> sqlite3.Row:
        row = self._db.execute(
            "SELECT * FROM rules WHERE rule_id=?",
            (rule_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown rule: {rule_id}")
        return row

    def _recompute_rule(self, rule_id: str) -> None:
        evidence_ids = self.evidence_for_rule(rule_id)
        counts = self._outcome_counts(evidence_ids)
        current = self._rule_row(rule_id)
        confidence = self._posterior_confidence(counts, prior=float(current["confidence"]))
        evidence_count = sum(counts.values())
        status = current["status"]
        if status == "active" and not self._activation_allowed(evidence_count, confidence):
            status = "candidate"
        with self._db:
            self._db.execute(
                """
                UPDATE rules SET confidence=?, status=?, updated_at=?,
                    success_count=?, failure_count=?, partial_count=?, unknown_count=?
                WHERE rule_id=?
                """,
                (
                    confidence,
                    status,
                    _now(),
                    counts["success"],
                    counts["failure"],
                    counts["partial"],
                    counts["unknown"],
                    rule_id,
                ),
            )

    @staticmethod
    def _row_to_rule(row: sqlite3.Row) -> ReasoningRule:
        return ReasoningRule(
            title=row["title"],
            trigger=row["trigger_text"],
            strategy=row["strategy"],
            rationale=row["rationale"],
            tags=tuple(json.loads(row["tags_json"])),
            confidence=float(row["confidence"]),
            status=row["status"],
            rule_id=row["rule_id"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            success_count=int(row["success_count"]),
            failure_count=int(row["failure_count"]),
            partial_count=int(row["partial_count"]),
            unknown_count=int(row["unknown_count"]),
            supersedes=row["supersedes"],
        )


class ContrastiveDistiller:
    """Deterministic baseline distiller for success/failure pairs."""

    @staticmethod
    def distill(
        success: Experience,
        failure: Experience,
        *,
        title: Optional[str] = None,
        tags: Sequence[str] = (),
    ) -> ReasoningRule:
        success.validate()
        failure.validate()
        if success.outcome != "success":
            raise ValueError("success experience must have outcome='success'")
        if failure.outcome != "failure":
            raise ValueError("failure experience must have outcome='failure'")

        trigger_terms = sorted(_tokens(success.task) & _tokens(failure.task))
        trigger = (
            "When handling tasks involving " + ", ".join(trigger_terms[:8])
            if trigger_terms
            else f"When handling tasks similar to: {success.task[:160]}"
        )
        strategy = (
            f"Prefer the verified successful action: {success.action}. "
            f"Avoid the failed action pattern: {failure.action}."
        )
        rationale = (
            f"Contrastive evidence: success observed '{success.observation[:240]}'; "
            f"failure observed '{failure.observation[:240]}'."
        )
        return ReasoningRule(
            title=title or f"Contrastive strategy: {success.task[:80]}",
            trigger=trigger,
            strategy=strategy,
            rationale=rationale,
            tags=tuple(tags),
            confidence=0.5,
            status="candidate",
        )


class ExperienceScaler:
    """Memory-aware test-time scaling (MaTTS-style) orchestration primitive.

    The organ does not execute candidate actions itself. It produces a bounded
    candidate budget informed by memory quality; callers remain responsible for
    policy checks, sandboxing, execution, and verification.
    """

    def __init__(
        self,
        bank: ReasoningBank,
        *,
        min_candidates: int = 2,
        max_candidates: int = 8,
    ) -> None:
        if min_candidates < 1 or max_candidates < min_candidates:
            raise ValueError("invalid candidate bounds")
        self.bank = bank
        self.min_candidates = min_candidates
        self.max_candidates = max_candidates

    def plan(self, task: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        memories = self.bank.retrieve(
            task,
            context=context,
            limit=self.max_candidates,
            include_candidates=True,
        )
        if not memories:
            budget = self.max_candidates
            reason = "no relevant procedural memory; maximize bounded exploration"
        else:
            best = memories[0]
            uncertainty = 1.0 - best.rule.confidence
            novelty = 1.0 - best.relevance
            exploration = 0.65 * uncertainty + 0.35 * novelty
            span = self.max_candidates - self.min_candidates
            budget = self.min_candidates + int(round(span * exploration))
            budget = max(self.min_candidates, min(self.max_candidates, budget))
            reason = (
                f"memory-aware budget from confidence={best.rule.confidence:.3f}, "
                f"relevance={best.relevance:.3f}"
            )

        return {
            "candidate_budget": budget,
            "reason": reason,
            "retrieved_rule_ids": [m.rule.rule_id for m in memories],
            "retrieved": [
                {
                    "rule_id": m.rule.rule_id,
                    "title": m.rule.title,
                    "status": m.rule.status,
                    "confidence": m.rule.confidence,
                    "relevance": m.relevance,
                    "overlap_terms": list(m.overlap_terms),
                }
                for m in memories
            ],
        }
