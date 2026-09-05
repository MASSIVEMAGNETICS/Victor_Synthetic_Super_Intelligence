import tempfile
import unittest
from pathlib import Path

from victor_os.reasoning_bank import (
    ContrastiveDistiller,
    Experience,
    ExperienceScaler,
    ReasoningBank,
)


class ReasoningBankTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "rb.sqlite3"
        self.bank = ReasoningBank(
            self.db,
            min_evidence_to_activate=2,
            min_confidence_to_activate=0.60,
        )

    def tearDown(self):
        self.bank.close()
        self.tmp.cleanup()

    def _seed_pair(self):
        success = Experience(
            task="deploy production api with tls",
            action="verify dns then tls then health endpoint",
            outcome="success",
            observation="HTTPS health endpoint returned 200",
            evidence={"http_status": 200, "tls": True},
            score=1.0,
            episode_id="ep-success",
        )
        failure = Experience(
            task="deploy production api with tls",
            action="declare deployment complete before tls verification",
            outcome="failure",
            observation="API hostname failed HTTPS validation",
            evidence={"tls": False},
            score=0.0,
            episode_id="ep-failure",
        )
        self.bank.record_experience(success)
        self.bank.record_experience(failure)
        return success, failure

    def test_failure_and_success_are_both_preserved(self):
        success, failure = self._seed_pair()
        self.assertEqual(self.bank.get_experience(success.experience_id).outcome, "success")
        self.assertEqual(self.bank.get_experience(failure.experience_id).outcome, "failure")
        receipt = self.bank.verify_integrity()
        self.assertTrue(receipt["ok"])
        self.assertEqual(receipt["experiences"], 2)

    def test_contrastive_rule_is_provenanced_and_retrievable(self):
        success, failure = self._seed_pair()
        rule = ContrastiveDistiller.distill(
            success, failure, tags=("deployment", "tls", "verification")
        )
        rule_id = self.bank.create_rule(
            rule,
            evidence_ids=(success.experience_id, failure.experience_id),
        )
        self.assertFalse(self.bank.promote_rule(rule_id))
        retrieved = self.bank.retrieve(
            "verify tls before production deployment",
            include_candidates=True,
        )
        self.assertTrue(retrieved)
        self.assertEqual(retrieved[0].rule.rule_id, rule_id)
        self.assertEqual(
            set(self.bank.evidence_for_rule(rule_id)),
            {success.experience_id, failure.experience_id},
        )

    def test_rule_can_activate_after_positive_evidence(self):
        success, failure = self._seed_pair()
        rule = ContrastiveDistiller.distill(success, failure, tags=("deployment",))
        rule_id = self.bank.create_rule(
            rule, evidence_ids=(success.experience_id, failure.experience_id)
        )
        second_success = Experience(
            task="deploy production api",
            action="verify dns tls assets api transaction path",
            outcome="success",
            observation="all production gates passed",
            evidence={"checks": 5},
            score=1.0,
        )
        self.bank.record_experience(second_success)
        self.bank.attach_evidence(rule_id, second_success.experience_id)
        self.assertTrue(self.bank.promote_rule(rule_id))
        self.assertEqual(self.bank.get_rule(rule_id).status, "active")
        active = self.bank.retrieve("production tls deployment verification")
        self.assertTrue(active)

    def test_revision_deprecates_old_rule_without_losing_evidence(self):
        success, failure = self._seed_pair()
        rule_id = self.bank.create_rule(
            ContrastiveDistiller.distill(success, failure),
            evidence_ids=(success.experience_id, failure.experience_id),
        )
        new_id = self.bank.revise_rule(
            rule_id,
            strategy="Verify DNS -> TLS -> health -> assets -> API -> transaction.",
        )
        self.assertEqual(self.bank.get_rule(rule_id).status, "deprecated")
        self.assertEqual(self.bank.get_rule(new_id).supersedes, rule_id)
        self.assertEqual(
            set(self.bank.evidence_for_rule(new_id)),
            {success.experience_id, failure.experience_id},
        )

    def test_scaler_expands_budget_when_memory_is_weak_or_missing(self):
        scaler = ExperienceScaler(self.bank, min_candidates=2, max_candidates=8)
        no_memory = scaler.plan("novel task never seen before")
        self.assertEqual(no_memory["candidate_budget"], 8)

        success, failure = self._seed_pair()
        rule_id = self.bank.create_rule(
            ContrastiveDistiller.distill(success, failure, tags=("deployment",)),
            evidence_ids=(success.experience_id, failure.experience_id),
        )
        weak = scaler.plan("deployment tls verification")
        self.assertGreaterEqual(weak["candidate_budget"], 2)
        self.assertLessEqual(weak["candidate_budget"], 8)
        self.assertIn(rule_id, weak["retrieved_rule_ids"])


if __name__ == "__main__":
    unittest.main()
