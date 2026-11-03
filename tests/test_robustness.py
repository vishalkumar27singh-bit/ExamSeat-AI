import numpy as np
import pytest

from image_processing import Transform, image_fingerprint, image_quality, load_image
from ocr_engine import Detection, OCREngine, extract_tokens
from pipeline import Analysis, add_enhanced_pass, all_numbers_table, search
from seat_locator import SeatLocator, VLine, find_room_headings


def _det(text, x=100, y=30, width=120, height=24):
    poly = np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
                    dtype=np.float32)
    return Detection(text, 0.95, poly)


@pytest.mark.parametrize("polygon", [
    None,
    [],
    [[1, 2, 3]],
    [[1, 2], [3, 4], [5, 6]],
    [[0, 0], [1, 0], [2, 0], [3, 0]],
    [[0, 0], [10, 0], [10, float("nan")], [0, 10]],
    [["bad", 0], [10, 0], [10, 10], [0, 10]],
])
def test_invalid_polygons_are_unverified_instead_of_crashing(polygon):
    locator = SeatLocator(np.full((100, 100, 3), 255, np.uint8), [])

    result = locator.locate(polygon)

    assert not result.verified
    assert result.room is None
    assert result.checks["valid_polygon"] is False
    assert any("invalid coordinates" in warning for warning in result.warnings)


def test_locator_skips_malformed_ocr_detections(caplog):
    image = np.full((100, 100, 3), 255, np.uint8)
    detections = [
        _det("2401100681"),
        Detection("bad polygon", 0.9, None),
        Detection("bad score", float("nan"), _det("204").poly),
        Detection("", 0.9, _det("204").poly),
    ]

    locator = SeatLocator(image, detections)

    assert [d.text for d in locator.dets] == ["2401100681"]
    assert "Ignoring malformed OCR detection" in caplog.text


@pytest.mark.parametrize("value", [
    None,
    np.empty((0, 4, 2), dtype=np.uint8),
    np.zeros((20, 20, 2), dtype=np.uint8),
    np.zeros((20, 20, 3), dtype=np.float32),
])
def test_load_image_rejects_empty_or_malformed_arrays(value):
    with pytest.raises(ValueError):
        load_image(value)


def test_grayscale_and_bgra_inputs_are_normalized_to_bgr():
    gray = load_image(np.zeros((12, 14), dtype=np.uint8))
    bgra = load_image(np.zeros((12, 14, 4), dtype=np.uint8))

    assert gray.shape == (12, 14, 3)
    assert bgra.shape == (12, 14, 3)


def test_ocr_engine_skips_malformed_results_and_keeps_valid_detection():
    class Backend:
        name = "test"

        def run(self, _img):
            return [
                (np.array([[1, 1], [20, 1], [20, 12], [1, 12]]), "204", 0.9),
                (None, "bad polygon", 0.9),
                (np.zeros((4, 2)), "degenerate", 0.9),
                (np.array([[1, 1], [20, 1], [20, 12], [1, 12]]), "bad score", float("nan")),
                (np.array([[1, 1], [20, 1], [20, 12], [1, 12]]), None, 0.9),
            ]

    engine = OCREngine.__new__(OCREngine)
    engine._b = Backend()

    result = engine.run(np.zeros((30, 30, 3), dtype=np.uint8))

    assert len(result) == 1
    assert result[0].text == "204"
    assert result[0].poly.shape == (4, 2)


def test_ocr_engine_treats_empty_backend_output_as_no_detections():
    class Backend:
        name = "test"

        def run(self, _img):
            return []

    engine = OCREngine.__new__(OCREngine)
    engine._b = Backend()

    assert engine.run(np.zeros((30, 30, 3), dtype=np.uint8)) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Room No: 204", "204"),
        ("Room Number: 204", "204"),
        ("ROOM-204", "204"),
        ("Room: 204", "204"),
        ("Room No: ATR-205", "ATR-205"),
    ],
)
def test_room_labels_extract_only_the_room_value(text, expected):
    headings = find_room_headings([_det(text)])

    assert [heading.room for heading in headings] == [expected]


def test_separate_room_label_and_number_are_supported_but_bare_rolls_are_not():
    label = _det("Room No:", x=10)
    value = _det("204", x=150)

    assert [heading.room for heading in find_room_headings([label, value])] == ["204"]
    assert find_room_headings([value]) == []


def test_multiple_explicit_room_numbers_remain_distinct():
    detections = [
        _det("Room No: 204", x=50, y=20),
        _det("Room Number: 305", x=50, y=300),
    ]

    assert [heading.room for heading in find_room_headings(detections)] == ["204", "305"]


@pytest.mark.parametrize(("title_x", "idx", "expected_idx"), [(500, 0, 0), (300, 3, 5)])
def test_completed_grid_includes_each_inferred_wall(title_x, idx, expected_idx):
    locator = SeatLocator(np.full((300, 900, 3), 255, np.uint8), [])
    locator.u = 10.0
    locator.wn = 60.0
    locator.titles = [np.array([title_x, 150.0])]
    lines = [VLine(i, 0.0, float(x), 0.0, 300.0, 300.0) for i, x in enumerate(range(200, 700, 100))]

    completed = locator._complete_chain(
        {"lines": lines, "idx": idx, "total": 4, "note": ""},
        cx=400.0,
        cy=200.0,
    )

    assert completed["total"] == 6
    assert len(completed["lines"]) == 7
    assert completed["idx"] == expected_idx
    assert 0 <= completed["idx"] < completed["total"]
    assert all(line.virtual for line in completed["lines"] if line.id < 0)


def test_empty_number_table_has_stable_columns_and_is_sortable():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    analysis = Analysis(
        original=image,
        work=image,
        to_work=Transform.identity(),
        detections=[],
        engine_name="test",
        engine_note=None,
        quality=image_quality(image),
        fingerprint=image_fingerprint(image),
    )

    table = all_numbers_table(analysis)

    assert table.empty
    assert "warnings" in table.columns
    assert "reg_no" in table.columns


def test_bundled_blurry_photo_keeps_enhanced_pass_when_it_recovers_numbers(engine):
    image = load_image("samples/photo_001_blurry.jpg")
    original = engine.run(image, "test-original")
    before = sum(token.kind == "full" for token in extract_tokens(original))
    analysis = Analysis(
        original=image,
        work=image,
        to_work=Transform.identity(),
        detections=original,
        engine_name=engine.name,
        engine_note=engine.fallback_reason,
        quality=image_quality(image),
        fingerprint=image_fingerprint(image),
    )

    add_enhanced_pass(analysis, engine)

    after = sum(token.kind == "full" for token in extract_tokens(analysis.detections))
    assert before > 0
    assert after > before
    assert any("improved" in note for note in analysis.notes)


def test_tilted_bundled_photo_has_a_non_crashing_all_numbers_table(engine):
    image = load_image("samples/photo_001_tilted.jpg")
    original = engine.run(image, "test-tilted")
    analysis = Analysis(
        original=image,
        work=image,
        to_work=Transform.identity(),
        detections=original,
        engine_name=engine.name,
        engine_note=engine.fallback_reason,
        quality=image_quality(image),
        fingerprint=image_fingerprint(image),
    )

    table = all_numbers_table(analysis)
    result = search(analysis, engine, "2401100681", retry_enhanced=False)

    assert len(table) >= 90
    assert table["reg_no"].notna().all()
    assert table["warnings"].notna().all()
    assert result.status == "FOUND"
    assert [(match.room, match.column, match.row) for match in result.matches] == [("ATR-205", "C", 1)]
    assert result.matches[0].seat.verified
