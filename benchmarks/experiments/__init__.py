"""
Translation Quality Experiments

This package contains tools for benchmarking translation quality
with and without constraint-aware agents.
"""

from .evaluator import TranslationEvaluator
from .baseline import BaselineTranslator
from .runner import ExperimentRunner, TestCase, ExperimentResults
from .reporter import ComparisonReporter
from .utils import (
    load_lyrics_from_json,
    sample_test_cases,
    create_test_suite,
    save_test_suite,
    load_test_suite,
)

__all__ = [
    "TranslationEvaluator",
    "BaselineTranslator",
    "ExperimentRunner",
    "ComparisonReporter",
    "TestCase",
    "ExperimentResults",
    "load_lyrics_from_json",
    "sample_test_cases",
    "create_test_suite",
    "save_test_suite",
    "load_test_suite",
]
