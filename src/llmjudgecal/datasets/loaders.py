"""CSV/JSONL loaders that build validated Dataset objects from user files.

Column names in the user's file rarely match our schema's field names, so
every loader accepts an optional `column_map` (canonical field name -> actual
column name in the file). Every error raised here is a `DatasetLoadError`
that names the source path, the 1-indexed row, and the offending field, so a
user can jump straight to the bad row without reading a pydantic traceback.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from llmjudgecal.datasets.schema import Dataset, PairwiseItem, Response, ScoreItem

PAIRWISE_REQUIRED_COLUMNS = (
    "id",
    "prompt",
    "response_a_id",
    "response_a_text",
    "response_b_id",
    "response_b_text",
    "human_label",
    "criterion",
)
PAIRWISE_OPTIONAL_COLUMNS = (
    "response_a_source_model",
    "response_a_source_family",
    "response_b_source_model",
    "response_b_source_family",
    "metadata",
)

SCORE_REQUIRED_COLUMNS = (
    "id",
    "prompt",
    "response_id",
    "response_text",
    "human_score",
    "scale_min",
    "scale_max",
    "criterion",
)
SCORE_OPTIONAL_COLUMNS = ("response_source_model", "response_source_family", "metadata")


class DatasetLoadError(ValueError):
    """Raised when a dataset file cannot be parsed into our schema.

    Example:
        >>> try:
        ...     raise DatasetLoadError("missing required column 'id'", path=Path("x.csv"), row=2)
        ... except DatasetLoadError as exc:
        ...     "row 2" in str(exc)
        True
    """

    def __init__(
        self,
        message: str,
        *,
        path: Path,
        row: int | None = None,
        field: str | None = None,
    ) -> None:
        location = str(path)
        if row is not None:
            location += f", row {row}"
        if field is not None:
            location += f", field '{field}'"
        super().__init__(f"{location}: {message}")
        self.path = path
        self.row = row
        self.field = field


def _resolve_column(column_map: Mapping[str, str], canonical: str) -> str:
    return column_map.get(canonical, canonical)


def _get_required(
    row: Mapping[str, Any],
    column_map: Mapping[str, str],
    canonical: str,
    *,
    path: Path,
    row_num: int,
) -> Any:
    actual_key = _resolve_column(column_map, canonical)
    if actual_key not in row:
        raise DatasetLoadError(
            f"missing required column '{actual_key}'", path=path, row=row_num, field=canonical
        )
    value = row[actual_key]
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise DatasetLoadError("required field is empty", path=path, row=row_num, field=canonical)
    return value


def _get_optional(row: Mapping[str, Any], column_map: Mapping[str, str], canonical: str) -> Any:
    actual_key = _resolve_column(column_map, canonical)
    value = row.get(actual_key)
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    return value


def _parse_metadata(raw: Any, *, path: Path, row_num: int) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise DatasetLoadError(
                f"metadata is not valid JSON: {exc}", path=path, row=row_num, field="metadata"
            ) from exc
        if not isinstance(parsed, dict):
            raise DatasetLoadError(
                "metadata must be a JSON object", path=path, row=row_num, field="metadata"
            )
        return parsed
    raise DatasetLoadError(
        "metadata must be a JSON object or a JSON object string",
        path=path,
        row=row_num,
        field="metadata",
    )


def _parse_float(raw: Any, *, path: Path, row_num: int, field: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise DatasetLoadError(
            f"'{raw}' is not a valid number", path=path, row=row_num, field=field
        ) from exc


def _format_validation_error(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error["loc"])
        parts.append(f"{loc}: {error['msg']}" if loc else str(error["msg"]))
    return "; ".join(parts)


def _iter_csv_rows(path: Path) -> Iterator[tuple[int, dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        yield from enumerate(reader, start=2)


def _iter_jsonl_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as fh:
        for row_num, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetLoadError(f"invalid JSON: {exc}", path=path, row=row_num) from exc
            if not isinstance(row, dict):
                raise DatasetLoadError("each line must be a JSON object", path=path, row=row_num)
            yield row_num, row


def _build_pairwise_item(
    row: Mapping[str, Any], row_num: int, path: Path, column_map: Mapping[str, str]
) -> PairwiseItem:
    item_id = _get_required(row, column_map, "id", path=path, row_num=row_num)
    prompt = _get_required(row, column_map, "prompt", path=path, row_num=row_num)
    a_id = _get_required(row, column_map, "response_a_id", path=path, row_num=row_num)
    a_text = _get_required(row, column_map, "response_a_text", path=path, row_num=row_num)
    b_id = _get_required(row, column_map, "response_b_id", path=path, row_num=row_num)
    b_text = _get_required(row, column_map, "response_b_text", path=path, row_num=row_num)
    human_label = _get_required(row, column_map, "human_label", path=path, row_num=row_num)
    criterion = _get_required(row, column_map, "criterion", path=path, row_num=row_num)
    metadata_raw = _get_optional(row, column_map, "metadata")
    metadata = _parse_metadata(metadata_raw, path=path, row_num=row_num)

    try:
        response_a = Response(
            id=str(a_id),
            text=str(a_text),
            source_model=_get_optional(row, column_map, "response_a_source_model"),
            source_family=_get_optional(row, column_map, "response_a_source_family"),
        )
        response_b = Response(
            id=str(b_id),
            text=str(b_text),
            source_model=_get_optional(row, column_map, "response_b_source_model"),
            source_family=_get_optional(row, column_map, "response_b_source_family"),
        )
        return PairwiseItem(
            id=str(item_id),
            prompt=str(prompt),
            response_a=response_a,
            response_b=response_b,
            human_label=human_label,
            criterion=str(criterion),
            metadata=metadata,
        )
    except ValidationError as exc:
        raise DatasetLoadError(_format_validation_error(exc), path=path, row=row_num) from exc


def _build_score_item(
    row: Mapping[str, Any], row_num: int, path: Path, column_map: Mapping[str, str]
) -> ScoreItem:
    item_id = _get_required(row, column_map, "id", path=path, row_num=row_num)
    prompt = _get_required(row, column_map, "prompt", path=path, row_num=row_num)
    r_id = _get_required(row, column_map, "response_id", path=path, row_num=row_num)
    r_text = _get_required(row, column_map, "response_text", path=path, row_num=row_num)
    human_score_raw = _get_required(row, column_map, "human_score", path=path, row_num=row_num)
    scale_min_raw = _get_required(row, column_map, "scale_min", path=path, row_num=row_num)
    scale_max_raw = _get_required(row, column_map, "scale_max", path=path, row_num=row_num)
    criterion = _get_required(row, column_map, "criterion", path=path, row_num=row_num)
    metadata_raw = _get_optional(row, column_map, "metadata")
    metadata = _parse_metadata(metadata_raw, path=path, row_num=row_num)

    human_score = _parse_float(human_score_raw, path=path, row_num=row_num, field="human_score")
    scale_min = _parse_float(scale_min_raw, path=path, row_num=row_num, field="scale_min")
    scale_max = _parse_float(scale_max_raw, path=path, row_num=row_num, field="scale_max")

    try:
        response = Response(
            id=str(r_id),
            text=str(r_text),
            source_model=_get_optional(row, column_map, "response_source_model"),
            source_family=_get_optional(row, column_map, "response_source_family"),
        )
        return ScoreItem(
            id=str(item_id),
            prompt=str(prompt),
            response=response,
            human_score=human_score,
            scale_min=scale_min,
            scale_max=scale_max,
            criterion=str(criterion),
            metadata=metadata,
        )
    except ValidationError as exc:
        raise DatasetLoadError(_format_validation_error(exc), path=path, row=row_num) from exc


def _build_dataset(
    items: Iterator[PairwiseItem] | Iterator[ScoreItem],
    *,
    mode: Literal["pairwise", "score"],
    path: Path,
    name: str | None,
    version: str,
    license: str,
    source_url: str | None,
) -> Dataset:
    items_tuple = tuple(items)
    if not items_tuple:
        raise DatasetLoadError("file contains no data rows", path=path)
    try:
        return Dataset(
            name=name or path.stem,
            version=version,
            mode=mode,
            items=items_tuple,  # type: ignore[arg-type]
            license=license,
            source_url=source_url or str(path),
        )
    except ValidationError as exc:
        raise DatasetLoadError(_format_validation_error(exc), path=path) from exc


def load_pairwise_csv(
    path: str | Path,
    column_map: Mapping[str, str] | None = None,
    *,
    name: str | None = None,
    version: str = "1.0",
    license: str = "unknown",
    source_url: str | None = None,
) -> Dataset:
    """Load a pairwise comparison dataset from a CSV file.

    Example:
        >>> from pathlib import Path
        >>> p = Path("examples/pairwise_sample.jsonl")  # doctest: +SKIP
        >>> load_pairwise_csv(p).mode  # doctest: +SKIP
        'pairwise'
    """
    path = Path(path)
    column_map = column_map or {}
    items = (
        _build_pairwise_item(row, row_num, path, column_map)
        for row_num, row in _iter_csv_rows(path)
    )
    return _build_dataset(
        items,
        mode="pairwise",
        path=path,
        name=name,
        version=version,
        license=license,
        source_url=source_url,
    )


def load_pairwise_jsonl(
    path: str | Path,
    column_map: Mapping[str, str] | None = None,
    *,
    name: str | None = None,
    version: str = "1.0",
    license: str = "unknown",
    source_url: str | None = None,
) -> Dataset:
    """Load a pairwise comparison dataset from a JSONL file, streaming line by line.

    Example:
        >>> from pathlib import Path
        >>> p = Path("examples/pairwise_sample.jsonl")  # doctest: +SKIP
        >>> load_pairwise_jsonl(p).mode  # doctest: +SKIP
        'pairwise'
    """
    path = Path(path)
    column_map = column_map or {}
    items = (
        _build_pairwise_item(row, row_num, path, column_map)
        for row_num, row in _iter_jsonl_rows(path)
    )
    return _build_dataset(
        items,
        mode="pairwise",
        path=path,
        name=name,
        version=version,
        license=license,
        source_url=source_url,
    )


def load_score_csv(
    path: str | Path,
    column_map: Mapping[str, str] | None = None,
    *,
    name: str | None = None,
    version: str = "1.0",
    license: str = "unknown",
    source_url: str | None = None,
) -> Dataset:
    """Load a single-answer scoring dataset from a CSV file.

    Example:
        >>> from pathlib import Path
        >>> p = Path("examples/score_sample.csv")  # doctest: +SKIP
        >>> load_score_csv(p).mode  # doctest: +SKIP
        'score'
    """
    path = Path(path)
    column_map = column_map or {}
    items = (
        _build_score_item(row, row_num, path, column_map) for row_num, row in _iter_csv_rows(path)
    )
    return _build_dataset(
        items,
        mode="score",
        path=path,
        name=name,
        version=version,
        license=license,
        source_url=source_url,
    )


def load_score_jsonl(
    path: str | Path,
    column_map: Mapping[str, str] | None = None,
    *,
    name: str | None = None,
    version: str = "1.0",
    license: str = "unknown",
    source_url: str | None = None,
) -> Dataset:
    """Load a single-answer scoring dataset from a JSONL file, streaming line by line.

    Example:
        >>> from pathlib import Path
        >>> p = Path("examples/score_sample.csv")  # doctest: +SKIP
        >>> load_score_jsonl(p).mode  # doctest: +SKIP
        'score'
    """
    path = Path(path)
    column_map = column_map or {}
    items = (
        _build_score_item(row, row_num, path, column_map) for row_num, row in _iter_jsonl_rows(path)
    )
    return _build_dataset(
        items,
        mode="score",
        path=path,
        name=name,
        version=version,
        license=license,
        source_url=source_url,
    )


_LOADERS: dict[tuple[str, str], Callable[..., Dataset]] = {
    ("pairwise", ".csv"): load_pairwise_csv,
    ("pairwise", ".jsonl"): load_pairwise_jsonl,
    ("score", ".csv"): load_score_csv,
    ("score", ".jsonl"): load_score_jsonl,
}


def load_dataset(
    path: str | Path,
    mode: Literal["pairwise", "score"],
    column_map: Mapping[str, str] | None = None,
    *,
    name: str | None = None,
    version: str = "1.0",
    license: str = "unknown",
    source_url: str | None = None,
) -> Dataset:
    """Load a dataset, dispatching to the right loader by mode and file suffix.

    Example:
        >>> from pathlib import Path
        >>> p = Path("examples/score_sample.csv")  # doctest: +SKIP
        >>> load_dataset(p, mode="score").mode  # doctest: +SKIP
        'score'
    """
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        loader = _LOADERS[(mode, suffix)]
    except KeyError:
        raise DatasetLoadError(
            f"unsupported file format '{suffix}' for mode '{mode}'", path=path
        ) from None
    return loader(
        path,
        column_map,
        name=name,
        version=version,
        license=license,
        source_url=source_url,
    )
