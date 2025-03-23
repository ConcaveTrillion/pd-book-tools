from pd_book_tools.book_ocr.ocr_word import OCRWord
from pd_book_tools.geometry import BoundingBox, Point


def test_ocrword_to_dict():
    bounding_box = BoundingBox(top_left=Point(0, 0), bottom_right=Point(10, 5))
    word = OCRWord(text="test", bounding_box=bounding_box, ocr_confidence=0.9)
    expected_dict = {
        "text": "test",
        "bounding_box": bounding_box.to_dict(),
        "ocr_confidence": 0.9,
    }
    assert word.to_dict() == expected_dict


def test_ocrword_from_dict():
    bounding_box_dict = {
        "top_left": {"x": 0, "y": 0},
        "bottom_right": {"x": 10, "y": 5},
    }
    word_dict = {
        "text": "test",
        "bounding_box": bounding_box_dict,
        "ocr_confidence": 0.9,
    }
    word = OCRWord.from_dict(word_dict)
    assert word.text == "test"
    assert word.bounding_box == BoundingBox.from_dict(bounding_box_dict)
    assert word.ocr_confidence == 0.9
