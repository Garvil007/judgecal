"""Command-line entry point for JudgeCal.

Example:
    >>> from typer.testing import CliRunner
    >>> runner = CliRunner()
    >>> result = runner.invoke(app, ["--version"])
    >>> result.exit_code
    0
"""

from __future__ import annotations

import typer

from llmjudgecal import __version__

app = typer.Typer(name="judgecal", help="Audit the reliability of an LLM-as-judge.")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"judgecal {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the JudgeCal version and exit.",
    ),
) -> None:
    """JudgeCal CLI root command."""


if __name__ == "__main__":
    app()
