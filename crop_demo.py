# -*- encoding: utf-8 -*-
# ! python3


import json
from pathlib import Path

import numpy as np
from PIL import Image


def get_frames_xywh(src_image: Image, bboxes: np.array) -> Image:
    assert bboxes.shape[1] == 4
    bboxes = xywh2xyxy(bboxes, src_image.size)

    for box in bboxes:
        yield _crop(box, src_image)


def _crop(box, src_image):
    assert max(box[0], box[2]) <= src_image.size[0]
    assert max(box[1], box[3]) <= src_image.size[1]
    return src_image.crop(box)


def xywh2xyxy(x: np.array, im_size: dict) -> np.array:
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

    return denormalize(y, im_size)


def denormalize(y: np.array, im_size: dict) -> np.array:
    y[:, 0] = y[:, 0] * im_size[0]
    y[:, 2] = y[:, 2] * im_size[0]
    y[:, 1] = y[:, 1] * im_size[1]
    y[:, 3] = y[:, 3] * im_size[1]

    return y


def main():
    raw_response = Path("./example_response.json").read_text(encoding="utf-8")
    parsed_response = json.loads(raw_response)
    yolo_boxes = np.array(
        [single_response["yolo"] for single_response in parsed_response]
    )

    with Image.open("./example_image.jpg") as original_input_image:
        for i, extracted_bb in enumerate(
            get_frames_xywh(original_input_image, yolo_boxes)
        ):
            extracted_bb.save(f"out/bb_{i}_example_image.jpg")


if __name__ == "__main__":
    main()
