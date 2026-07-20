from triage_dataset import bucket_for


def test_bucket_for_reuses_the_manually_verified_cases():
    assert bucket_for(0.005, 50) == "sharp_candidates"
    assert bucket_for(1.5, 50) == "intentional_blur_candidates"
    assert bucket_for(0.04, 50) == "unintentional_blur_candidates"
    assert bucket_for(0.02, 50) == "borderline"
    assert bucket_for(None, 50) == "unknown"
