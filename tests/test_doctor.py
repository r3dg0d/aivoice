from aivoice.doctor import format_doctor, run_doctor


def test_doctor_offline():
    checks = run_doctor(check_network=False)
    assert any(c.name == "Voice library" for c in checks)
    text = format_doctor(checks)
    assert "aivoice doctor" in text
