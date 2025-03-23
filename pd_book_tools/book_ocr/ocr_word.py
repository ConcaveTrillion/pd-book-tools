from dataclasses import dataclass

from ..geometry import BoundingBox


@dataclass
class OCRWord:
    """Represents a single word (uninterrupted sequence of characters) detected by OCR"""

    text: str
    bounding_box: BoundingBox
    ocr_confidence: float

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary"""
        return {
            "text": self.text,
            "bounding_box": self.bounding_box.to_dict(),
            "ocr_confidence": self.ocr_confidence,
        }

    def from_dict(dict) -> "OCRWord":
        """Create OCRWord from dictionary"""
        return OCRWord(
            text=dict["text"],
            bounding_box=BoundingBox.from_dict(dict["bounding_box"]),
            ocr_confidence=dict["ocr_confidence"],
        )
