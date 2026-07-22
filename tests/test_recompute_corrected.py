from pathlib import Path

from recompute_corrected import load_eval_runs, load_label_sources, parse_reviewer_specs
from review_analysis import DEFAULT_ADJUDICATIONS, compute_accuracy, compute_agreement, parse_adjudications


def test_parse_reviewer_specs_rejects_duplicate_names():
    try:
        parse_reviewer_specs(["alice=one.csv", "alice=two.csv"])
    except ValueError as exc:
        assert "Duplicate reviewer name" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected duplicate reviewer names to raise")


def test_parse_adjudications_accepts_comma_separated_entries():
    adjudications = parse_adjudications(["DSC06207=Unintentional Blur,DSC05659=sharp"])
    assert adjudications == {
        "DSC06207": "unintentional_blur",
        "DSC05659": "sharp",
    }


def test_default_recompute_regression_numbers_match_repo_data():
    original_source, blind_source, adjudicated_source, additional_sources = load_label_sources(
        Path("results/blind_full_review.csv"),
        adjudications=dict(DEFAULT_ADJUDICATIONS),
        reviewers={},
    )
    assert additional_sources == []

    agreement = compute_agreement(original_source, blind_source)
    assert (agreement.correct, agreement.total) == (31, 42)

    eval_runs = load_eval_runs()
    assert compute_accuracy(eval_runs["C_final"], original_source).correct == 28
    assert compute_accuracy(eval_runs["C_final"], blind_source).correct == 30
    assert compute_accuracy(eval_runs["D_final"], original_source).correct == 29
    assert compute_accuracy(eval_runs["D_final"], blind_source).correct == 27
    assert compute_accuracy(eval_runs["D_step20"], original_source).correct == 33
    assert compute_accuracy(eval_runs["D_step20"], blind_source).correct == 35

    assert compute_accuracy(eval_runs["C_final"], adjudicated_source).correct == 32
    assert compute_accuracy(eval_runs["D_final"], adjudicated_source).correct == 29
    assert compute_accuracy(eval_runs["D_step20"], adjudicated_source).correct == 37
