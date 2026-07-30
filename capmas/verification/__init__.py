"""Predicate, freshness, precondition, and postcondition interfaces."""

from capmas.verification.predicates import PredicateBasedVerifier, PredicateRegistry
from capmas.verification.evidence import (
    VerifierEvidence,
    VerifierPredicateEvidence,
    attach_verifier_evidence,
    predicate_report_to_evidence,
    summarize_verifier_results,
)

__all__ = [
    "PredicateBasedVerifier",
    "PredicateRegistry",
    "VerifierEvidence",
    "VerifierPredicateEvidence",
    "attach_verifier_evidence",
    "predicate_report_to_evidence",
    "summarize_verifier_results",
]
