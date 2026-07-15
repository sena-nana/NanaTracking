from typer.testing import CliRunner

from nana_tracking.cli import app

runner = CliRunner()


def test_doctor_reports_python_and_providers() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert '"python": "3.14.' in result.stdout
    assert "onnxruntime_providers" in result.stdout


def test_cli_help_lists_workflows() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "benchmark-python" in result.stdout
    assert "verify-export" in result.stdout
    assert "evaluation" in result.stdout


def test_data_and_evaluation_contract_commands() -> None:
    data_result = runner.invoke(app, ["data", "validate", "examples/manifests/synthetic-v1.json"])
    evaluation_result = runner.invoke(
        app,
        [
            "evaluation",
            "validate-standard",
            "configs/evaluation/ntp-v1-standard.json",
        ],
    )
    assert data_result.exit_code == 0
    assert evaluation_result.exit_code == 0
