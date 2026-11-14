"""Core pydantic v2 data schema for judge-calibration datasets and outputs.

All models are frozen: once a Response, dataset item, verdict, or score is
constructed it represents a recorded fact (a human label, a judge's raw
output) and must never be mutated in place. Downstream metrics code
(agreement, bias, consistency) reads these records from multiple places;
immutability rules out a whole class of bugs where one consumer's edit
leaks into another's view of the same object.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

Verdict = Literal["A", "B", "tie"]


def _require_non_blank(value: str, field_name: str | None) -> str:
    """Raise ValueError if value is empty or whitespace-only.

    Example:
        >>> _require_non_blank("hi", "text")
        'hi'
    """
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty or whitespace-only")
    return value


class Response(BaseModel):
    """A single model response under evaluation.

    Example:
        >>> Response(id="r1", text="Paris is the capital of France.").text
        'Paris is the capital of France.'
    """

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    source_model: str | None = None
    source_family: str | None = None

    @field_validator("id", "text")
    @classmethod
    def _not_blank(cls, value: str, info: ValidationInfo) -> str:
        return _require_non_blank(value, info.field_name)


class PairwiseItem(BaseModel):
    """A pairwise comparison item with a human label.

    Example:
        >>> item = PairwiseItem(
        ...     id="p1", prompt="Which is better?",
        ...     response_a=Response(id="a", text="A"),
        ...     response_b=Response(id="b", text="B"),
        ...     human_label="A", criterion="overall",
        ... )
        >>> item.human_label
        'A'
    """

    model_config = ConfigDict(frozen=True)

    id: str
    prompt: str
    response_a: Response
    response_b: Response
    human_label: Verdict
    criterion: str
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("id", "prompt", "criterion")
    @classmethod
    def _not_blank(cls, value: str, info: ValidationInfo) -> str:
        return _require_non_blank(value, info.field_name)


class ScoreItem(BaseModel):
    """A single-answer scoring item with a human score on a bounded scale.

    Example:
        >>> item = ScoreItem(
        ...     id="s1", prompt="Rate this", response=Response(id="r1", text="ok"),
        ...     human_score=7.0, scale_min=1.0, scale_max=10.0, criterion="fluency",
        ... )
        >>> item.human_score
        7.0
    """

    model_config = ConfigDict(frozen=True)

    id: str
    prompt: str
    response: Response
    human_score: float
    scale_min: float
    scale_max: float
    criterion: str
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("id", "prompt", "criterion")
    @classmethod
    def _not_blank(cls, value: str, info: ValidationInfo) -> str:
        return _require_non_blank(value, info.field_name)

    @model_validator(mode="after")
    def _validate_scale(self) -> ScoreItem:
        if self.scale_min >= self.scale_max:
            raise ValueError("scale_min must be less than scale_max")
        if not (self.scale_min <= self.human_score <= self.scale_max):
            raise ValueError(
                f"human_score {self.human_score} out of range [{self.scale_min}, {self.scale_max}]"
            )
        return self


DatasetItem = PairwiseItem | ScoreItem


class Dataset(BaseModel):
    """A versioned collection of pairwise or scoring items with provenance.

    Example:
        >>> dataset = Dataset(
        ...     name="toy", version="1.0", mode="score",
        ...     items=(
        ...         ScoreItem(
        ...             id="s1", prompt="Rate this", response=Response(id="r1", text="ok"),
        ...             human_score=7.0, scale_min=1.0, scale_max=10.0, criterion="fluency",
        ...         ),
        ...     ),
        ...     license="MIT", source_url="https://example.com/toy",
        ... )
        >>> len(dataset.items)
        1
    """

    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    mode: Literal["pairwise", "score"]
    items: tuple[DatasetItem, ...]
    license: str
    source_url: str

    @field_validator("name", "version", "license", "source_url")
    @classmethod
    def _not_blank(cls, value: str, info: ValidationInfo) -> str:
        return _require_non_blank(value, info.field_name)

    @model_validator(mode="after")
    def _validate_items(self) -> Dataset:
        if not self.items:
            raise ValueError("items must not be empty")

        expected_type = PairwiseItem if self.mode == "pairwise" else ScoreItem
        for item in self.items:
            if not isinstance(item, expected_type):
                raise ValueError(
                    f"all items must be {expected_type.__name__} when mode='{self.mode}'"
                )

        ids = [item.id for item in self.items]
        duplicates = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate item ids found: {duplicates}")
        return self


class TokenUsage(BaseModel):
    """Token counts for a single judge call.

    Example:
        >>> TokenUsage(prompt_tokens=10, completion_tokens=5).completion_tokens
        5
    """

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


class JudgeVerdict(BaseModel):
    """A judge's pairwise verdict with provenance (raw output, cost, latency).

    `verdict` is `None` when `parsed` is False: the raw judge output could not
    be parsed into "A"/"B"/"tie", but we still keep the raw text for debugging
    and for consistency metrics (an unparseable rate is itself a signal).

    Example:
        >>> JudgeVerdict(
        ...     verdict="A", raw_output="A is better", parsed=True,
        ...     latency_ms=120.5, token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        ...     cost_usd=0.002,
        ... ).verdict
        'A'
    """

    model_config = ConfigDict(frozen=True)

    verdict: Verdict | None
    raw_output: str
    parsed: bool
    latency_ms: float = Field(ge=0)
    token_usage: TokenUsage
    cost_usd: float = Field(ge=0)

    @model_validator(mode="after")
    def _validate_parsed(self) -> JudgeVerdict:
        if self.parsed and self.verdict is None:
            raise ValueError("verdict must be set when parsed is True")
        return self


class JudgeScore(BaseModel):
    """A judge's numeric score with provenance (raw output, cost, latency).

    `score` is `None` when `parsed` is False, for the same reason as
    `JudgeVerdict.verdict`.

    Example:
        >>> JudgeScore(
        ...     score=8.5, raw_output="8.5/10", parsed=True,
        ...     latency_ms=90.0, token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        ...     cost_usd=0.001,
        ... ).score
        8.5
    """

    model_config = ConfigDict(frozen=True)

    score: float | None
    raw_output: str
    parsed: bool
    latency_ms: float = Field(ge=0)
    token_usage: TokenUsage
    cost_usd: float = Field(ge=0)

    @model_validator(mode="after")
    def _validate_parsed(self) -> JudgeScore:
        if self.parsed and self.score is None:
            raise ValueError("score must be set when parsed is True")
        return self
