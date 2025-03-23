import unittest
from sortedcontainers import SortedList
from pd_book_tools.book_ocr.ocr_line import OCRLine
from pd_book_tools.book_ocr.ocr_word import OCRWord
from pd_book_tools.geometry import BoundingBox


class TestOCRLine(unittest.TestCase):

    def setUp(self):

        self.words = SortedList(
            [
                OCRWord(
                    text="Hello",
                    bounding_box=BoundingBox.from_ltrb(0, 0, 10, 10),
                    ocr_confidence=0.9,
                ),
                OCRWord(
                    text="world",
                    bounding_box=BoundingBox.from_ltrb(11, 0, 20, 10),
                    ocr_confidence=0.8,
                ),
            ],
            key=lambda word: word.bounding_box.top_left.x,
        )
        self.bounding_box = BoundingBox.from_ltrb(0, 0, 20, 10)
        self.ocr_line = OCRLine(words=self.words, bounding_box=self.bounding_box)

    def test_text(self):
        self.assertEqual(self.ocr_line.text, "Hello world")

    def test_mean_ocr_confidence(self):
        self.assertAlmostEqual(self.ocr_line.mean_ocr_confidence, 0.85)

    def test_to_dict(self):
        ocr_line_dict = self.ocr_line.to_dict()
        self.assertEqual(ocr_line_dict["text"], "Hello world")
        self.assertAlmostEqual(ocr_line_dict["mean_ocr_confidence"], 0.85)
        self.assertEqual(len(ocr_line_dict["words"]), 2)

    def test_from_dict(self):
        ocr_line_dict = self.ocr_line.to_dict()
        new_ocr_line = OCRLine.from_dict(ocr_line_dict)
        self.assertEqual(new_ocr_line.text, "Hello world")
        self.assertAlmostEqual(new_ocr_line.mean_ocr_confidence, 0.85)


if __name__ == "__main__":
    unittest.main()
