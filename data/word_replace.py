import os
import json


def replace_in_structure(data):
    if isinstance(data, dict):
        return {k: replace_in_structure(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_in_structure(item) for item in data]
    elif isinstance(data, str):
        return data.replace("Draged", "Drag")
    return data


def process_directory(base_path="general_EN"):
    for app_dir in os.listdir(base_path):
        app_path = os.path.join(base_path, app_dir)
        if not os.path.isdir(app_path):
            continue

        for scene_dir in os.listdir(app_path):
            scene_path = os.path.join(app_path, scene_dir)
            if not os.path.isdir(scene_path):
                continue

            json_path = os.path.join(scene_path, "task_infos.json")
            if not os.path.exists(json_path):
                continue

            with open(json_path, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                f.seek(0)
                json.dump(replace_in_structure(data), f, ensure_ascii=False, indent=4)
                f.truncate()


if __name__ == "__main__":
    process_directory()