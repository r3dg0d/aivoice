from click.testing import CliRunner

from aivoice.cli import main


def test_help():
    r = CliRunner().invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "voice" in r.output.lower() or "aivoice" in r.output.lower() or "MeanVC" in r.output


def test_subcommand_helps():
    runner = CliRunner()
    for args in (
        ["devices", "--help"],
        ["live", "--help"],
        ["file", "--help"],
        ["virtualmic", "--help"],
        ["profiles", "--help"],
        ["profile", "--help"],
        ["benchmark", "--help"],
        ["models", "--help"],
        ["models", "list"],
    ):
        r = runner.invoke(main, list(args))
        assert r.exit_code == 0, (args, r.output)


def test_live_requires_consent():
    r = CliRunner().invoke(main, ["live", "--reference", "/tmp/nope.wav"])
    assert r.exit_code != 0


def test_models_list_mentions_license_gap():
    r = CliRunner().invoke(main, ["models", "list"])
    assert r.exit_code == 0
    assert "meanvc2" in r.output.lower()
    assert "LICENSE" in r.output or "Apache" in r.output or "gap" in r.output.lower()
