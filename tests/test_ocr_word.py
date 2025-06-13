import pytest

from pd_book_tools.geometry.bounding_box import BoundingBox
from pd_book_tools.ocr.word import Word


def test_word_to_dict(sample_word1):
    word_dict = sample_word1.to_dict()

    assert word_dict == {
        "type": "Word",
        "text": "word1",
        "bounding_box": {
            "top_left": {"x": 0, "y": 0},
            "bottom_right": {"x": 10, "y": 10},
        },
        "ocr_confidence": pytest.approx(0.9),
        "word_labels": [],
        "ground_truth_text": None,
        "ground_truth_bounding_box": None,
        "ground_truth_match_keys": {},
    }


def test_word_from_dict(sample_word1):
    bounding_box_dict = {
        "top_left": {"x": 0, "y": 0},
        "bottom_right": {"x": 10, "y": 10},
    }
    word_dict = {
        "type": "Word",
        "text": "test",
        "bounding_box": bounding_box_dict,
        "ocr_confidence": 0.99,
        "word_labels": ["label1"],
    }
    word = Word.from_dict(word_dict)

    assert word.text == "test"
    assert word.bounding_box == BoundingBox.from_dict(bounding_box_dict)
    assert word.ocr_confidence == pytest.approx(0.99)
    assert word.word_labels == ["label1"]
