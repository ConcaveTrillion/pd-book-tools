from dataclasses import dataclass, field
from typing import Any, Dict, List

from sortedcontainers import SortedList

from ..geometry import BoundingBox
from .ocr_block import OCRBlock
from .ocr_line import OCRLine
from .ocr_word import OCRWord


@dataclass
class OCRPage:
    """Represents a page (single or multiple "blocks") of OCR results"""

    width: int
    height: int
    page_index: int
    bounding_box: BoundingBox
    blocks: SortedList = field(
        default_factory=lambda: SortedList(
            key=lambda block: block.bounding_box.point_min.y
        )
    )

    def __post_init__(self):
        # If no bounding_box provided, compute from blocks
        if self.bounding_box is None and self.blocks:
            self.bounding_box = BoundingBox.union(
                [block.bounding_box for block in self.blocks]
            )

    @property
    def text(self) -> str:
        """Get the full text of the page, separating each block by double carriage returns"""
        return "\n\n".join(block.text for block in self.blocks)

    @property
    def words(self) -> List[OCRWord]:
        """Get flat list of all words on the page ordered by appearance within the blocks"""
        words = []
        for block in self.blocks:
            for line in block.lines:
                words.extend(line.words)
        return words

    @property
    def lines(self) -> List[OCRLine]:
        """Get flat list of all lines on the page"""
        lines = []
        for block in self.blocks:
            lines.extend(block.lines)
        return lines

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return {
            "blocks": [block.to_dict() for block in self.blocks],
            "width": self.width,
            "height": self.height,
            "page_index": self.page_index,
            "bounding_box": self.bounding_box.to_dict(),
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, dict: Dict[str, Any]) -> "OCRPage":
        """Create OCRPage from dictionary"""
        return cls(
            blocks=SortedList([OCRBlock.from_dict(block) for block in dict["blocks"]]),
            width=dict["width"],
            height=dict["height"],
            page_index=dict["page_index"],
            bounding_box=(
                BoundingBox.from_dict(dict["bounding_box"])
                if dict.get("bounding_box")
                else None
            ),
        )
