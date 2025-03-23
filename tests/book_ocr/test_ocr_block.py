import unittest
from sortedcontainers import SortedList
from pd_book_tools.book_ocr import OCRBlock, OCRLine, OCRWord
from pd_book_tools.geometry import BoundingBox, Point


class TestOCRBlock(unittest.TestCase):

    def setUp(self):
        self.line1 = OCRLine(
            words=SortedList(
                iterable=[
                    OCRWord(
                        text="First",
                        bounding_box=BoundingBox(Point(0, 0), Point(5, 5)),
                        ocr_confidence=0.9,
                    ),
                    OCRWord(
                        text="line",
                        bounding_box=BoundingBox(Point(5, 0), Point(10, 5)),
                        ocr_confidence=0.9,
                    ),
                ],
                key=lambda word: word.bounding_box.top_left.x,
            ),
            bounding_box=BoundingBox(Point(0, 0), Point(10, 10)),
        )
        self.line2 = OCRLine(
            bounding_box=BoundingBox(Point(0, 10), Point(10, 20)),
            words=SortedList(
                iterable=[
                    OCRWord(
                        text="Second",
                        bounding_box=BoundingBox(Point(0, 10), Point(5, 15)),
                        ocr_confidence=0.9,
                    ),
                    OCRWord(
                        text="line",
                        bounding_box=BoundingBox(Point(5, 10), Point(10, 15)),
                        ocr_confidence=0.9,
                    ),
                ],
                key=lambda word: word.bounding_box.top_left.x,
            ),
        )
        self.lines = SortedList(
            [self.line1, self.line2],
            key=lambda line: line.bounding_box.top_left.y,
        )
        self.bounding_box = BoundingBox(Point(0, 0), Point(10, 20))
        self.ocr_block = OCRBlock(lines=self.lines, bounding_box=self.bounding_box)

    def test_text_property(self):
        self.assertEqual(self.ocr_block.text, "First line\nSecond line")

    def test_to_dict(self):
        ocr_block_dict = self.ocr_block.to_dict()
        self.assertEqual(ocr_block_dict["text"], "First line\nSecond line")
        self.assertEqual(len(ocr_block_dict["lines"]), 2)
        self.assertEqual(ocr_block_dict["bounding_box"], self.bounding_box.to_dict())

    def test_from_dict(self):
        ocr_block_dict = self.ocr_block.to_dict()
        new_ocr_block = OCRBlock.from_dict(ocr_block_dict)
        self.assertEqual(new_ocr_block.text, self.ocr_block.text)
        self.assertEqual(new_ocr_block.bounding_box, self.ocr_block.bounding_box)
        self.assertEqual(len(new_ocr_block.lines), len(self.ocr_block.lines))


if __name__ == "__main__":
    unittest.main()
