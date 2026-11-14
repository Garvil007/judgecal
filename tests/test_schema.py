"""Known-answer tests for llmjudgecal.datasets.schema, written before the implementation."""

import pytest
from pydantic import ValidationError

from llmjudgecal.datasets.schema import (
    Dataset,
    JudgeScore,
    JudgeVerdict,
    PairwiseItem,
    Response,
    ScoreItem,
    TokenUsage,
)


def make_response(id_: str = "r1", text: str = "hello world") -> Response:
    return Response(id=id_, text=text)


def make_pairwise_item(id_: str = "p1", human_label: str = "A") -> PairwiseItem:
    return PairwiseItem(
        id=id_,
        prompt="Which is better?",
        response_a=make_response("a1", "Answer A"),
        response_b=make_response("b1", "Answer B"),
        human_label=human_label,
        criterion="overall",
    )


def make_score_item(id_: str = "s1", human_score: float = 7.0) -> ScoreItem:
    return ScoreItem(
        id=id_,
        prompt="Rate this answer",
        response=make_response("r1", "Some answer"),
        human_score=human_score,
        scale_min=1.0,
        scale_max=10.0,
        criterion="fluency",
    )


def make_token_usage() -> TokenUsage:
    return TokenUsage(prompt_tokens=10, completion_tokens=5)


class TestResponse:
    def test_accepts_valid_response(self) -> None:
        response = make_response()
        assert response.text == "hello world"
        assert response.source_model is None
        assert response.source_family is None

    def test_rejects_empty_text(self) -> None:
        with pytest.raises(ValidationError):
            Response(id="r1", text="")

    def test_rejects_whitespace_only_text(self) -> None:
        with pytest.raises(ValidationError):
            Response(id="r1", text="   ")

    def test_rejects_empty_id(self) -> None:
        with pytest.raises(ValidationError):
            Response(id="", text="hello")

    def test_is_frozen(self) -> None:
        response = make_response()
        with pytest.raises(ValidationError):
            response.text = "changed"


class TestPairwiseItem:
    def test_accepts_valid_item(self) -> None:
        item = make_pairwise_item()
        assert item.human_label == "A"
        assert item.metadata == {}

    def test_rejects_empty_prompt(self) -> None:
        with pytest.raises(ValidationError):
            PairwiseItem(
                id="p1",
                prompt="",
                response_a=make_response("a1"),
                response_b=make_response("b1"),
                human_label="A",
                criterion="overall",
            )

    def test_rejects_invalid_human_label(self) -> None:
        with pytest.raises(ValidationError):
            make_pairwise_item(human_label="C")

    def test_accepts_tie_label(self) -> None:
        item = make_pairwise_item(human_label="tie")
        assert item.human_label == "tie"

    def test_is_frozen(self) -> None:
        item = make_pairwise_item()
        with pytest.raises(ValidationError):
            item.human_label = "B"


class TestScoreItem:
    def test_accepts_valid_item(self) -> None:
        item = make_score_item()
        assert item.human_score == 7.0

    def test_rejects_score_above_scale_max(self) -> None:
        with pytest.raises(ValidationError):
            make_score_item(human_score=11.0)

    def test_rejects_score_below_scale_min(self) -> None:
        with pytest.raises(ValidationError):
            make_score_item(human_score=0.0)

    def test_rejects_inverted_scale(self) -> None:
        with pytest.raises(ValidationError):
            ScoreItem(
                id="s1",
                prompt="Rate this",
                response=make_response(),
                human_score=5.0,
                scale_min=10.0,
                scale_max=1.0,
                criterion="fluency",
            )

    def test_rejects_equal_scale_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ScoreItem(
                id="s1",
                prompt="Rate this",
                response=make_response(),
                human_score=5.0,
                scale_min=5.0,
                scale_max=5.0,
                criterion="fluency",
            )


class TestDataset:
    def test_accepts_valid_pairwise_dataset(self) -> None:
        dataset = Dataset(
            name="toy",
            version="1.0",
            mode="pairwise",
            items=(make_pairwise_item("p1"), make_pairwise_item("p2")),
            license="MIT",
            source_url="https://example.com/toy",
        )
        assert len(dataset.items) == 2

    def test_accepts_valid_score_dataset(self) -> None:
        dataset = Dataset(
            name="toy",
            version="1.0",
            mode="score",
            items=(make_score_item("s1"), make_score_item("s2")),
            license="MIT",
            source_url="https://example.com/toy",
        )
        assert len(dataset.items) == 2

    def test_rejects_empty_items(self) -> None:
        with pytest.raises(ValidationError):
            Dataset(
                name="toy",
                version="1.0",
                mode="pairwise",
                items=(),
                license="MIT",
                source_url="https://example.com/toy",
            )

    def test_rejects_duplicate_ids(self) -> None:
        with pytest.raises(ValidationError):
            Dataset(
                name="toy",
                version="1.0",
                mode="pairwise",
                items=(make_pairwise_item("dup"), make_pairwise_item("dup")),
                license="MIT",
                source_url="https://example.com/toy",
            )

    def test_rejects_mode_item_mismatch(self) -> None:
        with pytest.raises(ValidationError):
            Dataset(
                name="toy",
                version="1.0",
                mode="pairwise",
                items=(make_score_item("s1"),),
                license="MIT",
                source_url="https://example.com/toy",
            )


class TestJudgeVerdict:
    def test_accepts_parsed_verdict(self) -> None:
        verdict = JudgeVerdict(
            verdict="A",
            raw_output="A is better",
            parsed=True,
            latency_ms=120.5,
            token_usage=make_token_usage(),
            cost_usd=0.002,
        )
        assert verdict.verdict == "A"

    def test_accepts_unparsed_verdict_with_none(self) -> None:
        verdict = JudgeVerdict(
            verdict=None,
            raw_output="garbage output",
            parsed=False,
            latency_ms=120.5,
            token_usage=make_token_usage(),
            cost_usd=0.002,
        )
        assert verdict.verdict is None

    def test_rejects_parsed_true_with_no_verdict(self) -> None:
        with pytest.raises(ValidationError):
            JudgeVerdict(
                verdict=None,
                raw_output="garbage output",
                parsed=True,
                latency_ms=120.5,
                token_usage=make_token_usage(),
                cost_usd=0.002,
            )

    def test_rejects_negative_latency(self) -> None:
        with pytest.raises(ValidationError):
            JudgeVerdict(
                verdict="A",
                raw_output="A is better",
                parsed=True,
                latency_ms=-1.0,
                token_usage=make_token_usage(),
                cost_usd=0.002,
            )

    def test_rejects_negative_cost(self) -> None:
        with pytest.raises(ValidationError):
            JudgeVerdict(
                verdict="A",
                raw_output="A is better",
                parsed=True,
                latency_ms=120.5,
                token_usage=make_token_usage(),
                cost_usd=-0.1,
            )


class TestJudgeScore:
    def test_accepts_parsed_score(self) -> None:
        score = JudgeScore(
            score=8.5,
            raw_output="8.5/10",
            parsed=True,
            latency_ms=90.0,
            token_usage=make_token_usage(),
            cost_usd=0.001,
        )
        assert score.score == 8.5

    def test_rejects_parsed_true_with_no_score(self) -> None:
        with pytest.raises(ValidationError):
            JudgeScore(
                score=None,
                raw_output="unparseable",
                parsed=True,
                latency_ms=90.0,
                token_usage=make_token_usage(),
                cost_usd=0.001,
            )


class TestTokenUsage:
    def test_rejects_negative_tokens(self) -> None:
        with pytest.raises(ValidationError):
            TokenUsage(prompt_tokens=-1, completion_tokens=5)
