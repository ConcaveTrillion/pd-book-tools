import itertools
from dataclasses import dataclass, field
from enum import Enum
from logging import getLogger
from typing import Collection, List, Optional, Tuple, Union

from numpy import ndarray
from thefuzz.fuzz import ratio as fuzz_ratio

from pd_book_tools.geometry.bounding_box import BoundingBox
from pd_book_tools.ocr.word import Word

# Configure logging
logger = getLogger(__name__)


class BlockChildType(Enum):
    WORDS = "WORDS"
    BLOCKS = "BLOCKS"


class BlockCategory(Enum):
    BLOCK = "BLOCK"
    PARAGRAPH = "PARAGRAPH"
    LINE = "LINE"


@dataclass
class Block:
    """
    Represents a block of text as detected and split by OCR.

    A "block" can be a line of text, a paragraph, or a larger "block" or "region" of text.
    Inside it, there can either be child blocks or individual words (a "line" of text)

    Some OCR tools may not distinguish between blocks, paragraphs, or lines.
    Some may have blocks with no words at all.
    """

    _items: List[Union["Word", "Block"]]
    bounding_box: Optional[BoundingBox] = None
    child_type: Optional[BlockChildType] = BlockChildType.BLOCKS
    block_category: Optional[BlockCategory] = BlockCategory.BLOCK
    block_labels: Optional[list[str]] = None
    additional_block_attributes: Optional[dict] = None
    base_ground_truth_text: Optional[str] = None
    """
    The "base" ground truth text of the block, not having been matched up to individual words or lines.
    """

    # TODO: Override page sort order for multi-column page layouts
    override_page_sort_order: Optional[int] = field(default=None)

    unmatched_ground_truth_words: Optional[List[Tuple[int, str]]] = field(
        default_factory=list
    )
    """
    For lines of text only, this is a list of tuples of (index, word) for words that are inserted
    in the line and not matched to ground truth.
    The index is the location right before the unmatched word is inserted into the line.
    This is used for data labeling, model training, and evaluation purposes.
    """

    def __init__(
        self,
        items: Collection,
        bounding_box: Optional[BoundingBox] = None,
        child_type: Optional[BlockChildType] = BlockChildType.BLOCKS,
        block_category: Optional[BlockCategory] = BlockCategory.BLOCK,
        block_labels: Optional[list[str]] = None,
        override_page_sort_order: Optional[int] = None,
        unmatched_ground_truth_words: Optional[List[Tuple[int, str]]] = None,
        additional_block_attributes: Optional[dict] = None,
        base_ground_truth_text: Optional[str] = None,
    ):
        self.child_type = child_type
        self.block_category = block_category
        self.block_labels = block_labels
        self.items = items  # Use the setter for validation or processing
        if bounding_box:
            self.bounding_box = bounding_box
        elif self.items:
            self.bounding_box = BoundingBox.union(
                [item.bounding_box for item in self.items]
            )
        else:
            self.bounding_box = None
        self.override_page_sort_order = override_page_sort_order

        if unmatched_ground_truth_words:
            self.unmatched_ground_truth_words = unmatched_ground_truth_words
        else:
            self.unmatched_ground_truth_words = []
        logger.debug(
            "unmatched_ground_truth_words: %s", str(self.unmatched_ground_truth_words)
        )

        if additional_block_attributes:
            self.additional_block_attributes = additional_block_attributes
        else:
            self.additional_block_attributes = {}

        self.base_ground_truth_text = base_ground_truth_text

    def _sort_items(self):
        # TODO: Implement a more robust sorting mechanism.

        # Blocks should be sorted:
        # Header & Page Number
        # Left Sidenotes
        # Body Text
        #    Within Body Text, sort by:
        #    Blocks, top to bottom
        #      Blocks within Blocks (Columns), left to right
        #      Within Columns, sort by Paragraphs, top to bottom
        #        Within Paragraphs, sort by Lines, top to bottom
        #          Within Lines, sort by Words, left to right

        # Right Sidenotes
        if self.child_type == BlockChildType.WORDS:
            self._items.sort(
                key=lambda item: (
                    item.bounding_box.top_left.x
                    if item.bounding_box and item.bounding_box.top_left
                    else 0,
                    item.bounding_box.top_left.y
                    if item.bounding_box and item.bounding_box.top_left
                    else 0,
                ),
            )
        else:
            self._items.sort(
                key=lambda item: (
                    item.bounding_box.top_left.y
                    if item.bounding_box and item.bounding_box.top_left
                    else 0,
                    item.bounding_box.top_left.x
                    if item.bounding_box and item.bounding_box.top_left
                    else 0,
                ),
            )

    @property
    def items(self) -> List:
        """Returns a copy of the item list in this block"""
        self._sort_items()
        return self._items.copy()

    def recompute_bounding_box(self):
        """Recompute the bounding box of the block based on its items"""
        if not self.items:
            return
        self.bounding_box = BoundingBox.union(
            [item.bounding_box for item in self.items]
        )

    def add_item(self, item):
        """Add an item to the block"""
        if self.child_type == BlockChildType.WORDS:
            if not isinstance(item, Word):
                raise TypeError("Item must be of type Word")
        else:
            if not isinstance(item, Block):
                raise TypeError("Item must be of type Block")
        self._items.append(item)
        self._sort_items()
        self.recompute_bounding_box()

    def remove_item(self, item):
        """Remove an item from the block"""
        if item in self._items:
            self._items.remove(item)
            self._sort_items()
            self.recompute_bounding_box()
            logger.debug(f"Empty item removed. New text: {self.text[0:10]}...")
        else:
            raise ValueError("Item not found in block")

    def remove_ground_truth(self):
        """Remove the ground truth text from the block"""
        self.unmatched_ground_truth_words = []
        if self.child_type == BlockChildType.WORDS:
            word: Word
            for word in self.items:
                word.ground_truth_text = ""
                word.ground_truth_bounding_box = None
        else:
            for block in self.items:
                block.remove_ground_truth()
        logger.debug("Ground truth text removed from block")

    def remove_line_if_exists(self, line):
        """Remove a line from the page if it exists"""
        if self.child_type == BlockChildType.WORDS:
            return

        if line in self.lines:
            if line in self._items:
                self.remove_item(line)
            else:
                for block in self._items:
                    if isinstance(block, Word):
                        raise TypeError("Block must be of type Block")
                    block.remove_line_if_exists(line)
            logger.debug(f"Line {line.text[0:10]}... removed from block")
        else:
            logger.debug(f"Line {line.text[0:10]}... not found in block")

    def remove_empty_items(self):
        """Remove empty child blocks from the block."""
        if not self.items:
            return
        if self.child_type != BlockChildType.WORDS:
            item: Block
            for item in self.items:
                item.remove_empty_items()
                if not item.items:
                    self.remove_item(item)
                    logger.debug("Empty block removed")
        # Empty words are directly removed with remove_item, do not need to be handled here

    @items.setter
    def items(self, value):
        if not isinstance(value, Collection):
            raise TypeError("items must be a collection (e.g., list, tuple, set)")
        for item in value:
            if not hasattr(item, "bounding_box") or not isinstance(
                item.bounding_box, BoundingBox
            ):
                raise TypeError(
                    "Each item in items must have a bounding_box attribute of type BoundingBox"
                )
            if not isinstance(item, (Word, Block)):
                raise TypeError("Each item in items must be of type Word or Block")
        self._items = list(value)
        self._sort_items()
        self.recompute_bounding_box()

    @property
    def text(self) -> str:
        """Get the full text of the block.
        If child type is words, join text by spaces.
        Otherwise join text by carriage returns.
        This automatically adds additional CRs between blocks/paragraphs.
        """
        if self.child_type == BlockChildType.WORDS:
            return " ".join(item.text for item in self.items)
        elif self.block_category == BlockCategory.PARAGRAPH:
            return "\n".join(item.text for item in self.items)
        else:
            return "\n\n".join(item.text for item in self.items)

    @property
    def ground_truth_text(self) -> str:
        """Get the ground truth text of the words in the block.
        If child type is words, join text by spaces.
        Otherwise join text by carriage returns.
        This automatically adds additional CRs between blocks/paragraphs.
        """
        if self.child_type == BlockChildType.WORDS:
            # If the block is a line, use the ground truth text of the words
            matched_words = []
            for word in self.items:
                matched_words.append(word.ground_truth_text or "")

            # Also, add unmatched ground truth words to the text
            if self.unmatched_ground_truth_words:
                for unmatched_gt_word_idx, unmatched_gt_word in reversed(
                    self.unmatched_ground_truth_words
                ):
                    logger.debug(
                        f"Adding unmatched ground truth word '{unmatched_gt_word}' at index {unmatched_gt_word_idx}"
                    )
                    matched_words.insert(unmatched_gt_word_idx + 1, unmatched_gt_word)
                logger.debug(
                    f"Matched words after adding unmatched ground truth words: {matched_words}"
                )
            return " ".join(matched_words)
        elif self.block_category == BlockCategory.PARAGRAPH:
            return "\n".join(item.ground_truth_text for item in self.items)
        else:
            return "\n\n".join(item.ground_truth_text for item in self.items)

    @property
    def ground_truth_text_only_ocr(self) -> str:
        """Get the ground truth text of the words in the block.
        Only include words that have associated OCR text.
        If child type is words, join text by spaces.
        Otherwise join text by carriage returns.
        This automatically adds additional CRs between blocks/paragraphs.
        """
        if self.child_type == BlockChildType.WORDS:
            return " ".join(item.ground_truth_text_only_ocr for item in self.items)
        elif self.block_category == BlockCategory.PARAGRAPH:
            return "\n".join(item.ground_truth_text_only_ocr for item in self.items)
        else:
            return "\n\n".join(item.ground_truth_text_only_ocr for item in self.items)

    @property
    def ground_truth_exact_match(self) -> bool:
        """Check if the ground truth text of the block matches the text"""
        if self.child_type == BlockChildType.WORDS:
            return all(item.ground_truth_exact_match for item in self.items)
        else:
            return all(item.ground_truth_exact_match for item in self.items)

    @property
    def word_list(self) -> list[str]:
        """Get list of words in the block"""
        if self.child_type == BlockChildType.WORDS:
            return [item.text for item in self.items]
        else:
            return list(
                itertools.chain.from_iterable([item.word_list for item in self.items])
            )

    @property
    def words(self) -> list[Word]:
        """Get flat list of all words in the block"""
        if self.child_type == BlockChildType.WORDS:
            return list(self.items)
        else:
            return list(
                itertools.chain.from_iterable([item.words for item in self.items])
            )

    @property
    def lines(self) -> List["Block"]:
        """Flat list of all 'lines' in the block"""
        if self.child_type == BlockChildType.WORDS:
            return [self]
        else:
            return list(
                itertools.chain.from_iterable([item.lines for item in self.items])
            )

    @property
    def paragraphs(self) -> List["Block"]:
        """Flat list of all 'paragraphs' in the block"""
        if self.block_category == BlockCategory.PARAGRAPH:
            return [self]
        else:
            return list(
                itertools.chain.from_iterable([item.paragraphs for item in self.items])
            )

    # def compute_baseline_y(self):
    #     # If the block is a line, compute a baseline
    #     """Compute the baseline Y-coordinate for a line of text."""
    #     if self.block_category != BlockCategory.LINE:
    #         raise ValueError("Baseline can only be computed for lines of text.")

    #     # Collect the bottom edges of all words in the line
    #     bottom_edges = [item.bounding_box.bottom_right.y for item in self.items]

    #     if not bottom_edges:
    #         return None

    #     # Use the median to ignore descenders like 'p' and 'q'
    #     baseline_y = int(np.median(bottom_edges))
    #     return baseline_y

    def split_word(
        self,
        split_word_index: int,
        bbox_split_offset: float,
        character_split_index: int,
    ):
        """Split a word in the line into two parts and replace it with the new words"""
        logger.debug(
            f"Line Splitting word at index {split_word_index} with bbox_split_offset {bbox_split_offset} and character_split_index {character_split_index}"
        )

        if self.child_type != BlockChildType.WORDS:
            raise ValueError("Cannot split a word in a block of blocks")
        if split_word_index < 0 or split_word_index >= len(self.items):
            raise IndexError("Index out of range")
        word = self.items[split_word_index]
        if not isinstance(word, Word):
            raise TypeError("Item must be of type Word")

        word_1, word_2 = word.split(
            bbox_split_offset=bbox_split_offset,
            character_split_index=character_split_index,
        )
        logger.debug("Word Split. New words: %s, %s", word_1.text, word_2.text)
        self.remove_item(word)
        self.add_item(word_1)
        self.add_item(word_2)
        self.remove_empty_items()
        self.recompute_bounding_box()
        logger.debug(f"Line after split:\n{self.text}")

    def merge(self, block_to_merge: "Block"):
        """Merge another block into this one"""
        if self.child_type != block_to_merge.child_type:
            raise ValueError("Cannot merge blocks with different child types")
        if self.block_category != block_to_merge.block_category:
            raise ValueError("Cannot merge blocks with different block categories")
        if self.bounding_box and block_to_merge.bounding_box:
            self.bounding_box = BoundingBox.union(
                [self.bounding_box, block_to_merge.bounding_box]
            )
        else:
            self.bounding_box = block_to_merge.bounding_box
        self._items.extend(block_to_merge.items)
        self._sort_items()
        if (
            self.unmatched_ground_truth_words
            and block_to_merge.unmatched_ground_truth_words
        ):
            self.unmatched_ground_truth_words.extend(
                block_to_merge.unmatched_ground_truth_words
            )
        if self.block_labels is None:
            self.block_labels = block_to_merge.block_labels
        else:
            if block_to_merge.block_labels is None:
                return
            self.block_labels = list(
                set(self.block_labels).union(block_to_merge.block_labels)
            )
        self.recompute_bounding_box()

    def ocr_confidence_scores(self) -> list[float]:
        """Get a list of the OCR confidence scores of all nested words"""
        if not self.items:
            return []
        if self.child_type == BlockChildType.WORDS:
            return [item.ocr_confidence for item in self.items]
        else:
            return list(
                itertools.chain.from_iterable(
                    [item.ocr_confidence_scores() for item in self.items]
                )
            )

    def mean_ocr_confidence(self) -> float:
        """Get the mean of the OCR confidence score of all items"""
        scores = self.ocr_confidence_scores()
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def scale(self, width: int, height: int) -> "Block":
        """
        Return new block with scaled bounding box
        and scaled children to absolute pixel coordinates
        """
        return Block(
            items=[item.scale(width, height) for item in self.items],
            bounding_box=self.bounding_box.scale(width, height)
            if self.bounding_box
            else None,
            child_type=self.child_type,
            block_category=self.block_category,
            block_labels=self.block_labels,
        )

    def fuzz_score_against(self, ground_truth_text):
        """Scores a string as "matching" against a ground truth string

        TODO: Perhaps add loose scoring for curly quotes against straight quotes, and em-dashes against hyphens to count these as "closer" to gt

        Args:
            ground_truth_text (_type_): 'correct' text
        """
        return fuzz_ratio(self.text, ground_truth_text)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary"""
        return {
            "type": "Block",
            "child_type": self.child_type.value if self.child_type else None,
            "block_category": self.block_category.value
            if self.block_category
            else None,
            "block_labels": self.block_labels,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "items": [item.to_dict() for item in self.items] if self.items else [],
            "override_page_sort_order": self.override_page_sort_order,
            "unmatched_ground_truth_words": self.unmatched_ground_truth_words or [],
            "additional_block_attributes": self.additional_block_attributes or {},
            "base_ground_truth_text": self.base_ground_truth_text or "",
        }

    @classmethod
    def from_dict(cls, dict) -> "Block":
        """Create OCRBlock from dictionary"""
        if dict.get("child_type"):
            child_type = BlockChildType(dict["child_type"])
        else:
            child_type = BlockChildType.WORDS

        if child_type == BlockChildType.WORDS:
            items = [Word.from_dict(item) for item in dict["items"]]
        else:
            items = [Block.from_dict(item) for item in dict["items"]]

        return cls(
            items=items,
            bounding_box=BoundingBox.from_dict(dict["bounding_box"]),
            child_type=child_type,
            block_category=BlockCategory(dict.get("block_category")),
            block_labels=dict.get("block_labels", []),
            override_page_sort_order=dict.get("override_page_sort_order", None),
            unmatched_ground_truth_words=dict.get("unmatched_ground_truth_words", []),
            additional_block_attributes=dict.get("additional_block_attributes", {}),
            base_ground_truth_text=dict.get("base_ground_truth_text", None),
        )

    def refine_bounding_boxes(self, image: ndarray | None, padding_px: int = 0):
        logger.debug(f"Refining bounding boxes for block with {len(self.items)} items")
        if not self.items:
            self.bounding_box = None
            return
        if self.child_type == BlockChildType.WORDS:
            logger.debug(
                f"Refining bounding boxes for {len(self.items)} words in block"
            )
            for item in self.items:
                logger.debug(
                    f"Refining bounding box for item: {getattr(item, 'text', str(item))}"
                )
                if (
                    hasattr(item, "refine_bounding_box")
                    and callable(getattr(item, "refine_bounding_box", None))
                    and not isinstance(item, Block)
                ):
                    item.refine_bounding_box(image, padding_px=padding_px)
                else:
                    logger.critical(
                        f"Item '{getattr(item, 'text', str(item))}' does not have a refine_bounding_box method"
                    )
                    raise NotImplementedError(
                        "Item does not implement refine_bounding_box method"
                    )
        else:
            logger.debug(
                f"Refining bounding boxes for {len(self.items)} blocks in block"
            )
            item: Block
            for item in self.items:
                item.refine_bounding_boxes(image, padding_px=padding_px)
        self.recompute_bounding_box()
