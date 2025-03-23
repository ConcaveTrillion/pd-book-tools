from dataclasses import dataclass, field
from typing import Optional

from sortedcontainers import SortedList

from ..geometry import BoundingBox
from .ocr_word import OCRWord


@dataclass
class OCRLine:
    """Represents a "line" of text as detected by the OCR"""

    words: SortedList = field(
        default_factory=lambda: SortedList(
            key=lambda word: word.bounding_box.point_min.x
        )
    )
    bounding_box: Optional[BoundingBox] = None

    def __post_init__(self):
        # If no bounding_box provided, compute from bounding_boxes of the words
        if self.bounding_box is None and self.words:
            self.bounding_box = BoundingBox.union(
                [word.bounding_box for word in self.words]
            )

    @property
    def text(self) -> str:
        """Get the full text of the line, separated by single spaces"""
        return " ".join(word.text for word in self.words)

    @property
    def mean_ocr_confidence(self) -> float:
        """Get the mean of the OCR confidence score of all words"""
        if not self.words:
            return 0.0
        return sum(word.ocr_confidence for word in self.words) / len(self.words)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary"""
        return {
            "words": [word.to_dict() for word in self.words],
            "bounding_box": self.bounding_box.to_dict(),
            "text": self.text,
            "mean_ocr_confidence": self.mean_ocr_confidence,
        }

    def from_dict(dict) -> "OCRLine":
        """Create OCRLine from dictionary"""
        return OCRLine(
            words=SortedList(
                [OCRWord.from_dict(word) for word in dict["words"]],
                key=lambda word: word.bounding_box.top_left.x,
            ),
            bounding_box=BoundingBox.from_dict(dict["bounding_box"]),
        )
