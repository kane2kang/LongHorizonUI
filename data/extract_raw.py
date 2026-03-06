import os
import shutil
import json


def extract_raw_data():
    source_dir = "game_data"
    dest_dir = "game_data_raw"

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for scene_folder in os.listdir(source_dir):
        scene_path = os.path.join(source_dir, scene_folder)
        if not os.path.isdir(scene_path):
            continue
        print(f"Processing scene: {scene_folder}")
        scene_dest = os.path.join(dest_dir, scene_folder)
        if not os.path.exists(scene_dest):
            os.makedirs(scene_dest)
        screenshot_source = os.path.join(scene_path, "screenshot")
        screenshot_dest = os.path.join(scene_dest, "screenshot")

        if not os.path.exists(screenshot_dest):
            os.makedirs(screenshot_dest)
        if os.path.exists(screenshot_source) and os.path.isdir(screenshot_source):
            print("  Copying screenshots...")
            for image_file in os.listdir(screenshot_source):
                # 跳过包含highlight的文件
                if "highlight" in image_file:
                    print(f"    Skipping highlighted file: {image_file}")
                    continue

                source_file = os.path.join(screenshot_source, image_file)
                dest_file = os.path.join(screenshot_dest, image_file)
                shutil.copy2(source_file, dest_file)
                print(f"    Copied: {image_file}")
        action_file = os.path.join(scene_path, "actions.json")
        dest_action_file = os.path.join(scene_dest, "actions.json")

        if os.path.exists(action_file):
            print("  Copying action file...")
            shutil.copy2(action_file, dest_action_file)
            print(f"    Copied actions.json")
            clean_action_file(dest_action_file)
        else:
            print(f"  WARNING: Actions file not found in {scene_folder}")


def clean_action_file(action_file):
    """从actions.json文件中删除highlight字段"""
    try:
        with open(action_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            for action in data:
                if "highlight" in action:
                    del action["highlight"]
        elif isinstance(data, dict):
            if "actions" in data and isinstance(data["actions"], list):
                for action in data["actions"]:
                    if "highlight" in action:
                        del action["highlight"]

        with open(action_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            print("    Cleared highlight fields in actions.json")

    except Exception as e:
        print(f"    Error cleaning actions file: {str(e)}")


if __name__ == "__main__":
    extract_raw_data()
    print("All scene data extracted successfully!")