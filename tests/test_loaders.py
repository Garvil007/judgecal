"""Known-answer tests for llmjudgecal.datasets.loaders, written before the implementation."""

import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from llmjudgecal.datasets.loaders import (
    DatasetLoadError,
    load_dataset,
    load_pairwise_csv,
    load_pairwise_jsonl,
    load_score_csv,
    load_score_jsonl,
)

PAIRWISE_CSV_HEADER = (
    "id,prompt,response_a_id,response_a_text,response_b_id,response_b_text,human_label,criterion\n"
)

SCORE_CSV_HEADER = "id,prompt,response_id,response_text,human_score,scale_min,scale_max,criterion\n"


def write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


class TestLoadPairwiseCsv:
    def test_happy_path(self, tmp_path: Path) -> None:
        csv_text = (
            PAIRWISE_CSV_HEADER
            + "p1,Which is better?,a1,Answer A,b1,Answer B,A,overall\n"
            + "p2,Pick one,a2,Answer A2,b2,Answer B2,tie,overall\n"
        )
        path = write(tmp_path / "pairwise.csv", csv_text)

        dataset = load_pairwise_csv(path)

        assert dataset.mode == "pairwise"
        assert len(dataset.items) == 2
        first = dataset.items[0]
        assert first.id == "p1"
        assert first.prompt == "Which is better?"
        assert first.response_a.id == "a1"
        assert first.response_a.text == "Answer A"
        assert first.response_b.text == "Answer B"
        assert first.human_label == "A"
        assert first.criterion == "overall"

    def test_column_map_remaps_names(self, tmp_path: Path) -> None:
        csv_text = (
            "uid,question,a_id,a_text,b_id,b_text,label,crit\n"
            "p1,Which is better?,a1,Answer A,b1,Answer B,A,overall\n"
        )
        path = write(tmp_path / "pairwise.csv", csv_text)
        column_map = {
            "id": "uid",
            "prompt": "question",
            "response_a_id": "a_id",
            "response_a_text": "a_text",
            "response_b_id": "b_id",
            "response_b_text": "b_text",
            "human_label": "label",
            "criterion": "crit",
        }

        dataset = load_pairwise_csv(path, column_map=column_map)

        assert dataset.items[0].id == "p1"
        assert dataset.items[0].prompt == "Which is better?"

    def test_missing_column_raises_with_field(self, tmp_path: Path) -> None:
        csv_text = (
            "id,prompt,response_a_id,response_a_text,response_b_id,response_b_text,criterion\n"
            "p1,Which is better?,a1,Answer A,b1,Answer B,overall\n"
        )
        path = write(tmp_path / "pairwise.csv", csv_text)

        with pytest.raises(DatasetLoadError) as exc_info:
            load_pairwise_csv(path)

        message = str(exc_info.value)
        assert "human_label" in message
        assert "row 2" in message

    def test_bad_human_label_raises_with_row_number(self, tmp_path: Path) -> None:
        csv_text = (
            PAIRWISE_CSV_HEADER
            + "p1,Which is better?,a1,Answer A,b1,Answer B,A,overall\n"
            + "p2,Which is better?,a2,Answer A2,b2,Answer B2,C,overall\n"
        )
        path = write(tmp_path / "pairwise.csv", csv_text)

        with pytest.raises(DatasetLoadError) as exc_info:
            load_pairwise_csv(path)

        message = str(exc_info.value)
        assert "row 3" in message
        assert "human_label" in message

    def test_duplicate_ids_raises(self, tmp_path: Path) -> None:
        csv_text = (
            PAIRWISE_CSV_HEADER
            + "p1,Which is better?,a1,Answer A,b1,Answer B,A,overall\n"
            + "p1,Pick one,a2,Answer A2,b2,Answer B2,tie,overall\n"
        )
        path = write(tmp_path / "pairwise.csv", csv_text)

        with pytest.raises(DatasetLoadError) as exc_info:
            load_pairwise_csv(path)

        assert "duplicate" in str(exc_info.value).lower()

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = write(tmp_path / "pairwise.csv", PAIRWISE_CSV_HEADER)

        with pytest.raises(DatasetLoadError) as exc_info:
            load_pairwise_csv(path)

        assert "no data rows" in str(exc_info.value)

    def test_unicode_text_preserved(self, tmp_path: Path) -> None:
        csv_text = (
            PAIRWISE_CSV_HEADER
            + 'p1,"¿Qué es mejor?",a1,"日本語の回答 \U0001f600",'
            + "b1,Answer B,A,overall\n"
        )
        path = write(tmp_path / "pairwise.csv", csv_text)

        dataset = load_pairwise_csv(path)

        assert dataset.items[0].prompt == "¿Qué es mejor?"
        assert dataset.items[0].response_a.text == "日本語の回答 \U0001f600"


class TestLoadPairwiseJsonl:
    def test_happy_path(self, tmp_path: Path) -> None:
        rows = [
            {
                "id": "p1",
                "prompt": "Which is better?",
                "response_a_id": "a1",
                "response_a_text": "Answer A",
                "response_b_id": "b1",
                "response_b_text": "Answer B",
                "human_label": "A",
                "criterion": "overall",
            },
            {
                "id": "p2",
                "prompt": "Pick one",
                "response_a_id": "a2",
                "response_a_text": "Answer A2",
                "response_b_id": "b2",
                "response_b_text": "Answer B2",
                "human_label": "tie",
                "criterion": "overall",
            },
        ]
        path = tmp_path / "pairwise.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        dataset = load_pairwise_jsonl(path)

        assert len(dataset.items) == 2
        assert dataset.items[1].human_label == "tie"

    def test_invalid_json_line_raises_with_row_number(self, tmp_path: Path) -> None:
        path = tmp_path / "pairwise.jsonl"
        path.write_text('{"id": "p1", invalid}\n', encoding="utf-8")

        with pytest.raises(DatasetLoadError) as exc_info:
            load_pairwise_jsonl(path)

        assert "row 1" in str(exc_info.value)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = write(tmp_path / "pairwise.jsonl", "")

        with pytest.raises(DatasetLoadError) as exc_info:
            load_pairwise_jsonl(path)

        assert "no data rows" in str(exc_info.value)

    def test_metadata_dict_passthrough(self, tmp_path: Path) -> None:
        row = {
            "id": "p1",
            "prompt": "Which is better?",
            "response_a_id": "a1",
            "response_a_text": "Answer A",
            "response_b_id": "b1",
            "response_b_text": "Answer B",
            "human_label": "A",
            "criterion": "overall",
            "metadata": {"source": "synthetic"},
        }
        path = tmp_path / "pairwise.jsonl"
        path.write_text(json.dumps(row), encoding="utf-8")

        dataset = load_pairwise_jsonl(path)

        assert dataset.items[0].metadata == {"source": "synthetic"}


class TestLoadScoreCsv:
    def test_happy_path(self, tmp_path: Path) -> None:
        csv_text = (
            SCORE_CSV_HEADER
            + "s1,Rate this,r1,Some answer,7.5,1,10,fluency\n"
            + "s2,Rate this too,r2,Another answer,3,1,10,fluency\n"
        )
        path = write(tmp_path / "score.csv", csv_text)

        dataset = load_score_csv(path)

        assert dataset.mode == "score"
        assert len(dataset.items) == 2
        assert dataset.items[0].human_score == 7.5
        assert dataset.items[0].scale_min == 1.0
        assert dataset.items[0].scale_max == 10.0

    def test_score_out_of_scale_raises_with_row_number(self, tmp_path: Path) -> None:
        csv_text = (
            SCORE_CSV_HEADER
            + "s1,Rate this,r1,Some answer,7.5,1,10,fluency\n"
            + "s2,Rate this too,r2,Another answer,15,1,10,fluency\n"
        )
        path = write(tmp_path / "score.csv", csv_text)

        with pytest.raises(DatasetLoadError) as exc_info:
            load_score_csv(path)

        assert "row 3" in str(exc_info.value)

    def test_non_numeric_score_raises(self, tmp_path: Path) -> None:
        csv_text = SCORE_CSV_HEADER + "s1,Rate this,r1,Some answer,great,1,10,fluency\n"
        path = write(tmp_path / "score.csv", csv_text)

        with pytest.raises(DatasetLoadError) as exc_info:
            load_score_csv(path)

        message = str(exc_info.value)
        assert "row 2" in message
        assert "human_score" in message


class TestLoadScoreJsonl:
    def test_happy_path(self, tmp_path: Path) -> None:
        row = {
            "id": "s1",
            "prompt": "Rate this",
            "response_id": "r1",
            "response_text": "Some answer",
            "human_score": 7.5,
            "scale_min": 1,
            "scale_max": 10,
            "criterion": "fluency",
        }
        path = tmp_path / "score.jsonl"
        path.write_text(json.dumps(row), encoding="utf-8")

        dataset = load_score_jsonl(path)

        assert dataset.items[0].human_score == 7.5


class TestLoadDataset:
    def test_dispatches_pairwise_csv(self, tmp_path: Path) -> None:
        csv_text = PAIRWISE_CSV_HEADER + "p1,Which is better?,a1,Answer A,b1,Answer B,A,overall\n"
        path = write(tmp_path / "pairwise.csv", csv_text)

        dataset = load_dataset(path, mode="pairwise")

        assert dataset.mode == "pairwise"

    def test_dispatches_score_jsonl(self, tmp_path: Path) -> None:
        row = {
            "id": "s1",
            "prompt": "Rate this",
            "response_id": "r1",
            "response_text": "Some answer",
            "human_score": 7.5,
            "scale_min": 1,
            "scale_max": 10,
            "criterion": "fluency",
        }
        path = tmp_path / "score.jsonl"
        path.write_text(json.dumps(row), encoding="utf-8")

        dataset = load_dataset(path, mode="score")

        assert dataset.mode == "score"

    def test_unsupported_suffix_raises(self, tmp_path: Path) -> None:
        path = write(tmp_path / "data.txt", "irrelevant")

        with pytest.raises(DatasetLoadError):
            load_dataset(path, mode="pairwise")


safe_text = st.text(min_size=1, max_size=30).filter(lambda s: s.strip() != "")


class TestRoundtripProperty:
    @given(
        item_id=safe_text,
        prompt=safe_text,
        a_id=safe_text,
        a_text=safe_text,
        b_id=safe_text,
        b_text=safe_text,
        human_label=st.sampled_from(["A", "B", "tie"]),
        criterion=safe_text,
    )
    def test_pairwise_jsonl_roundtrip(
        self,
        item_id: str,
        prompt: str,
        a_id: str,
        a_text: str,
        b_id: str,
        b_text: str,
        human_label: str,
        criterion: str,
    ) -> None:
        row = {
            "id": item_id,
            "prompt": prompt,
            "response_a_id": a_id,
            "response_a_text": a_text,
            "response_b_id": b_id,
            "response_b_text": b_text,
            "human_label": human_label,
            "criterion": criterion,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "roundtrip.jsonl"
            path.write_text(json.dumps(row), encoding="utf-8")
            dataset = load_pairwise_jsonl(path)

        item = dataset.items[0]
        assert item.id == item_id
        assert item.prompt == prompt
        assert item.response_a.id == a_id
        assert item.response_a.text == a_text
        assert item.response_b.id == b_id
        assert item.response_b.text == b_text
        assert item.human_label == human_label
        assert item.criterion == criterion

    @given(
        item_id=safe_text,
        prompt=safe_text,
        r_id=safe_text,
        r_text=safe_text,
        criterion=safe_text,
        bounds=st.tuples(
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        ).filter(lambda t: t[0] < t[1]),
    )
    def test_score_jsonl_roundtrip(
        self,
        item_id: str,
        prompt: str,
        r_id: str,
        r_text: str,
        criterion: str,
        bounds: tuple[float, float],
    ) -> None:
        scale_min, scale_max = bounds
        human_score = (scale_min + scale_max) / 2
        row = {
            "id": item_id,
            "prompt": prompt,
            "response_id": r_id,
            "response_text": r_text,
            "human_score": human_score,
            "scale_min": scale_min,
            "scale_max": scale_max,
            "criterion": criterion,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "roundtrip.jsonl"
            path.write_text(json.dumps(row), encoding="utf-8")
            dataset = load_score_jsonl(path)

        item = dataset.items[0]
        assert item.id == item_id
        assert item.prompt == prompt
        assert item.response.id == r_id
        assert item.response.text == r_text
        assert item.human_score == human_score
        assert item.scale_min == scale_min
        assert item.scale_max == scale_max
        assert item.criterion == criterion
