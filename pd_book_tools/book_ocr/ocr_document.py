import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

from pandas import DataFrame
from sortedcontainers import SortedList

from ..geometry import BoundingBox
from .ocr_block import OCRBlock
from .ocr_line import OCRLine
from .ocr_page import OCRPage
from .ocr_word import OCRWord


@dataclass
class OCRDocument:
    """
    Represents single/multiple pages of OCR results from an OCR engine.
    Currently supports doctr and tesseract outputs.
    """

    source_lib: str = ""
    source_path: Optional[Path] = None
    pages: SortedList = field(
        default_factory=lambda: SortedList(key=lambda page: page.page_index)
    )

    def to_dict(self) -> Dict:
        """Convert to a JSON-serializable dictionary"""
        return {
            "source_lib": self.source_lib,
            "source_path": str(self.source_path),
            "pages": [page.to_dict() for page in self.pages],
        }

    def from_dict(dict) -> "OCRDocument":
        """Create OCRDocument from dictionary"""
        return OCRDocument(
            source_lib=dict["source_lib"],
            source_path=Path(dict["source_path"]),
            pages=SortedList(
                [OCRPage.from_dict(page) for page in dict["pages"]],
                key=lambda page: page.page_index,
            ),
        )

    def save_json(self, file_path: Union[str, Path]) -> None:
        """Save OCR results to JSON file"""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_doctr_output(
        cls,
        doctr_output: Dict,
        source_path: Union[str, Path],
    ) -> "OCRDocument":
        """Create OCRDocument from docTR output"""
        if isinstance(source_path, str):
            source_path = Path(source_path)

        result = cls(source_lib="doctr", source_path=source_path)

        for page_idx, page_data in enumerate(doctr_output.get("pages", [])):
            height, width = page_data.get("dimensions", (0, 0))
            blocks = SortedList(key=lambda block: block.bounding_box.point_min.y)
            for block_data in page_data.get("blocks", []):
                if block_data.get("geometry"):
                    block_bounding_box = BoundingBox.from_nested_float(
                        block_data["geometry"]
                    )

                lines = SortedList(key=lambda line: line.bounding_box.point_min.y)
                for line_data in block_data.get("lines", []):
                    if line_data.get("geometry"):
                        line_bounding_box = BoundingBox.from_nested_float(
                            line_data["geometry"]
                        )

                    words = SortedList(key=lambda word: word.bounding_box.point_min.x)
                    for word_data in line_data.get("words", []):
                        word = OCRWord(
                            text=word_data.get("value", ""),
                            bounding_box=BoundingBox.from_nested_float(
                                word_data["geometry"]
                            ),
                            confidence=word_data.get("confidence", 0.0),
                        )
                        words.add(word)

                    line = OCRLine(words=words, bounding_box=line_bounding_box)
                    lines.add(line)

                block = OCRBlock(lines=lines, bounding_box=block_bounding_box)
                blocks.add(block)

            page = OCRPage(
                page_index=page_idx, width=width, height=height, blocks=blocks
            )
            result.pages.add(page)

    @classmethod
    def from_tesseract(
        cls,
        tesseract_output: DataFrame,
        source_path: Union[str, Path],
    ) -> "OCRDocument":
        """Create OCRDocument from Tesseract output"""
        if isinstance(source_path, str):
            source_path = Path(source_path)

        result = cls(source_lib="tesseract", source_path=source_path)

        page_filter = tesseract_output["level"] == 1.0
        page_filtered = tesseract_output.where(page_filter).dropna(how="all")
        for page_idx, page_row in page_filtered.itertuples():
            left, top, width, height = (
                page_row["left"],
                page_row["top"],
                page_row["width"],
                page_row["height"],
            )
            page_bounding_box = BoundingBox.from_ltwh(left, top, width, height)

            blocks = SortedList(key=lambda block: block.bounding_box.point_min.y)
            block_filter = (
                tesseract_output["level"] == 2.0
                and tesseract_output["page_num"] == page_idx
            )
            block_filtered = tesseract_output.where(block_filter).dropna(how="all")
            for block_idx, block_row in block_filtered.itertuples():
                left, top, width, height = (
                    block_row["left"],
                    block_row["top"],
                    block_row["width"],
                    block_row["height"],
                )
                block_bounding_box = BoundingBox.from_ltwh(left, top, width, height)

                # Skip tesseract "paragraphs" (level 3)

                lines = SortedList(key=lambda line: line.bounding_box.point_min.y)
                line_filter = (
                    tesseract_output["level"] == 4.0
                    and tesseract_output["page_num"] == page_idx
                    and tesseract_output["block_num"] == block_idx
                )
                line_filtered = tesseract_output.where(line_filter).dropna(how="all")
                for line_idx, line_row in line_filtered.itertuples():
                    left, top, width, height = (
                        line_row["left"],
                        line_row["top"],
                        line_row["width"],
                        line_row["height"],
                    )
                    line_bounding_box = BoundingBox.from_ltwh(left, top, width, height)

                    words = []
                    word_filter = (
                        tesseract_output["level"] == 5.0
                        and tesseract_output["page_num"] == page_idx
                        and tesseract_output["block_num"] == block_idx
                        and tesseract_output["line_num"] == line_idx
                    )
                    word_filtered = tesseract_output.where(word_filter).dropna(
                        how="all"
                    )
                    for word_row in word_filtered.itertuples():
                        left, top, width, height = (
                            word_row["left"],
                            word_row["top"],
                            word_row["width"],
                            word_row["height"],
                        )
                        word_bounding_box = BoundingBox.from_ltwh(
                            left, top, width, height
                        )

                        word = OCRWord(
                            text=word_row["text"],
                            bounding_box=word_bounding_box,
                            confidence=word_row["conf"],
                        )
                        words.add(word)

                    line = OCRLine(words=words, bounding_box=line_bounding_box)
                    lines.add(line)

                block = OCRBlock(lines=lines, bounding_box=block_bounding_box)
                blocks.add(block)

            page = OCRPage(
                page_index=page_idx,
                width=width,
                height=height,
                blocks=blocks,
                page_bounding_box=page_bounding_box,
            )
            result.pages.add(page)
