from __future__ import annotations

from ivy.cli import doctor


def test_doctor_is_json_serialisable():
    report = doctor()
    assert report["package"] == "hesperus-ivy"
    assert "frameworks" in report
    assert "cuda" in report
