"""Trivial smoke tests for the JudgeCal CLI."""

from typer.testing import CliRunner

from llmjudgecal import __version__
from llmjudgecal.cli import app

runner = CliRunner()


def test_version_flag_prints_version_and_exits_zero() -> None:
    # Arrange
    # Act
    result = runner.invoke(app, ["--version"])

    # Assert
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_true_is_true() -> None:
    # Arrange / Act / Assert
    assert True
