import pdb
import sys
import os
import cv2

sys.path.append(".")


def test_android_context():
    import adbutils
    from LonghorizonAgent.system.android_context import AndroidContext, AndroidContextConfig
    from LonghorizonAgent.common import utils
    devices = adbutils.adb.device_list()
    android_device = devices[0].info["serialno"]
    context_config = AndroidContextConfig(
        device_id=android_device,
        use_perception=True,
        use_ocr=True,
        use_ocr_rec=False,
        perception_description_type="md",
        highlight_type="normal",
        highlight_grid_num=24,
        screenshot_save_dir=f"./tmp/screenshots/android_{android_device}"
    )
    context = AndroidContext(config=context_config)
    cur_state = context.update_state()
    with open(os.path.splitext(cur_state.screenshot_path)[0] + ".md", "w") as fw:
        fw.write(cur_state.perception_description)


def test_computer_context():
    import adbutils
    from LonghorizonAgent.system.computer_context import ComputerContext, ComputerContextConfig
    from LonghorizonAgent.common import utils

    context_config = ComputerContextConfig(
        use_perception=True,
        use_ocr=True,
        use_ocr_rec=False,
        perception_description_type="md",
        highlight_type="normal",
        highlight_grid_num=24,
        screenshot_save_dir=f"./tmp/screenshots/computer"
    )
    context = ComputerContext(config=context_config)
    cur_state = context.update_state()
    with open(os.path.splitext(cur_state.screenshot_path)[0] + ".md", "w") as fw:
        fw.write(cur_state.perception_description)


if __name__ == '__main__':
    # test_android_context()
    test_computer_context()
