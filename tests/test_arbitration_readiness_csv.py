from src.diagnostics.arbitration_readiness_csv import rate



def test_rate_handles_empty_denominator() -> None:
    assert rate(1, 0) == 0.0
    assert rate(1, 2) == 0.5
