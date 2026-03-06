import pdb
import sys

sys.path.append(".")


def test_draw():
    import cv2
    import numpy as np
    import os

    from LonghorizonAgent.common import utils

    image_path = "./tmp/screenshots/android_device/8bebb893-ce0a-46cd-b1fa-074bd20742db.png"
    image = cv2.imread(image_path)

    image_draw = utils.add_grid_with_numbers(image, font_path="assets/fonts/arial.ttf", grid_num=16)

    result_dir = os.path.join("results", "grid_draw")
    os.makedirs(result_dir, exist_ok=True)
    img_save_path = os.path.join(result_dir, f"{os.path.basename(image_path)}")
    cv2.imwrite(img_save_path, image_draw)
    print(img_save_path)


if __name__ == '__main__':
    test_draw()
