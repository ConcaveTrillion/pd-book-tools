from dataclasses import dataclass, field
from typing import Optional

from sortedcontainers import SortedList

from ..geometry import BoundingBox
from .ocr_line import OCRLine


@dataclass
class OCRBlock:
    """Represents a block of text (multiple paragraphs) as detected and split by OCR."""

    lines: SortedList[OCRLine] = field(
        default_factory=lambda: SortedList(
            key=lambda line: line.bounding_box.point_min.y
        )
    )
    bounding_box: Optional[BoundingBox] = None

    def __post_init__(self):
        # If no bounding_box provided, compute from lines
        if self.bounding_box is None and self.lines:
            self.bounding_box = BoundingBox.union(
                [line.bounding_box for line in self.lines]
            )

    @property
    def text(self) -> str:
        """Get the full text of the block, joined by carriage returns"""
        return "\n".join(line.text for line in self.lines)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary"""
        return {
            "lines": [line.to_dict() for line in self.lines],
            "bounding_box": self.bounding_box.to_dict(),
            "text": self.text,
        }

    def from_dict(dict) -> "OCRBlock":
        """Create OCRBlock from dictionary"""
        return OCRBlock(
            lines=SortedList(
                [OCRLine.from_dict(line) for line in dict["lines"]],
                key=lambda line: line.bounding_box.top_left.y,
            ),
            bounding_box=BoundingBox.from_dict(dict["bounding_box"]),
        )
