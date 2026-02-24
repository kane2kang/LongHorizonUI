import pdb
import sys
import os
import cv2

sys.path.append(".")


def test_android_controller():
    from LonghorizonAgent.controller.android_controller import AndroidController

    controller = AndroidController()
    available_actions = controller.registry.get_prompt_description()
    print(available_actions)


if __name__ == '__main__':
    test_android_controller()
