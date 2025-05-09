from base64 import b64encode

from numpy import ndarray

from ..geometry import BoundingBox
from ..image_processing.cv2_processing import encode_bgr_image_as_png
from ..ocr.block import Block
from ..ocr.word import Word


def get_cropped_image(
    img: ndarray, bounding_box: BoundingBox
) -> tuple[ndarray, str, str]:
    h, w = img.shape[:2]
    # Get the bounding box of the word
    x1, y1, x2, y2 = bounding_box.scale(w, h).to_ltrb()
    # Crop the image to the bounding box
    cropped_img = img[y1:y2, x1:x2]
    # Encode the cropped image as PNG
    encoded_img = encode_bgr_image_as_png(cropped_img)
    b64_encoded_string = b64encode(encoded_img).decode("utf-8")
    data_src_string = f"data:image/png;base64,{b64_encoded_string}"
    return encoded_img, b64_encoded_string, data_src_string


def get_cropped_word_image(img: ndarray, word: Word):
    return get_cropped_image(img, word.bounding_box)


def get_cropped_block_image(img: ndarray, line: Block):
    return get_cropped_image(img, line.bounding_box)
