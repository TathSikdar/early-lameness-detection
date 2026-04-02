from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any


@dataclass
class EarTagDetectionRecord:
    record_id: str
    frame_index: int
    bbox_xyxy: list[int]
    confidence: float
    class_id: int


def _validate_record(record: dict[str, Any]) -> None:
    required = {"record_id", "frame_index", "bbox_xyxy", "confidence", "class_id"}
    missing = required.difference(record)
    if missing:
        raise ValueError(f"Ear-tag record missing fields: {sorted(missing)}")

    if not isinstance(record["record_id"], str) or not record["record_id"].strip():
        raise ValueError("record_id must be a non-empty string")

    if not isinstance(record["frame_index"], int) or record["frame_index"] < 0:
        raise ValueError("frame_index must be a non-negative int")

    bbox = record["bbox_xyxy"]
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(v, int) for v in bbox):
        raise ValueError("bbox_xyxy must be a list of four ints [x1, y1, x2, y2]")

    if not isinstance(record["confidence"], (int, float)):
        raise ValueError("confidence must be numeric")

    if not isinstance(record["class_id"], int):
        raise ValueError("class_id must be int")

    if "last_row" in record:
        _validate_last_row_payload(record["last_row"])

    if "ocr" in record:
        _validate_ocr_payload(record["ocr"])


def _validate_last_row_payload(payload: dict[str, Any]) -> None:
    required = {"bbox_xyxy", "confidence"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"last_row missing fields: {sorted(missing)}")

    bbox = payload["bbox_xyxy"]
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(v, int) for v in bbox):
        raise ValueError("last_row.bbox_xyxy must be a list of four ints [x1, y1, x2, y2]")

    if not isinstance(payload["confidence"], (int, float)):
        raise ValueError("last_row.confidence must be numeric")


def _validate_ocr_payload(payload: dict[str, Any]) -> None:
    required = {"predicted_cow_id", "confidence"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"ocr missing fields: {sorted(missing)}")

    if not isinstance(payload["predicted_cow_id"], str):
        raise ValueError("ocr.predicted_cow_id must be a string")

    if not isinstance(payload["confidence"], (int, float)):
        raise ValueError("ocr.confidence must be numeric")


def _build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    predicted_ids = {
        r["ocr"]["predicted_cow_id"]
        for r in records
        if "ocr" in r and isinstance(r["ocr"].get("predicted_cow_id"), str) and r["ocr"]["predicted_cow_id"].strip()
    }

    return {
        "frames_with_detections": len({r["frame_index"] for r in records}),
        "total_detections": len(records),
        "last_row_records": sum(1 for r in records if "last_row" in r),
        "ocr_records": sum(1 for r in records if "ocr" in r),
        "unique_predicted_cow_ids": sorted(predicted_ids),
    }


def build_ear_tag_metadata(
    session_id: str,
    cow_temp_id: str,
    source_video_path: str,
    records: list[dict[str, Any]],
    view: str = "side",
) -> dict[str, Any]:
    if not session_id:
        raise ValueError("session_id is required")
    if not cow_temp_id:
        raise ValueError("cow_temp_id is required")
    if not source_video_path:
        raise ValueError("source_video_path is required")

    for record in records:
        _validate_record(record)

    return {
        "session_id": session_id,
        "cow_temp_id": cow_temp_id,
        "view": view,
        "source_video_path": source_video_path,
        "records": records,
        "summary": _build_summary(records),
    }


def validate_ear_tag_metadata_document(document: dict[str, Any]) -> None:
    required = {
        "session_id",
        "cow_temp_id",
        "view",
        "source_video_path",
        "records",
        "summary",
    }
    missing = required.difference(document)
    if missing:
        raise ValueError(f"Ear-tag metadata missing top-level fields: {sorted(missing)}")

    if not isinstance(document["records"], list):
        raise ValueError("records must be a list")

    for record in document["records"]:
        _validate_record(record)


def enrich_record_with_last_row(
    document: dict[str, Any],
    record_id: str,
    bbox_xyxy: list[int],
    confidence: float,
) -> dict[str, Any]:
    for record in document["records"]:
        if record.get("record_id") == record_id:
            record["last_row"] = {
                "bbox_xyxy": bbox_xyxy,
                "confidence": float(confidence),
            }
            _validate_record(record)
            document["summary"] = _build_summary(document["records"])
            return document

    raise KeyError(f"record_id not found: {record_id}")


def enrich_record_with_ocr(
    document: dict[str, Any],
    record_id: str,
    predicted_cow_id: str,
    confidence: float,
) -> dict[str, Any]:
    for record in document["records"]:
        if record.get("record_id") == record_id:
            record["ocr"] = {
                "predicted_cow_id": predicted_cow_id,
                "confidence": float(confidence),
            }
            _validate_record(record)
            document["summary"] = _build_summary(document["records"])
            return document

    raise KeyError(f"record_id not found: {record_id}")


def load_json_document(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload


def write_json_document(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path
