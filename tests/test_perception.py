import pdb
import sys

sys.path.append(".")


def test_rapid_ocr():
    import cv2
    import os
    from rapidocr_onnxruntime import RapidOCR
    from LonghorizonAgent.common import vis

    engine = RapidOCR()

    image_path = "data/examples/web.png"
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result, elapse = engine(image_bgr, use_cls=False)
    image_draw = vis.visualize_ocr_results(image_bgr, result)
    print(elapse)
    save_dir = f"./results/rapid_ocr_tests"
    os.makedirs(save_dir, exist_ok=True)
    image_save_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(image_save_path, image_draw)
    print(image_save_path)


def test_surya_ocr():
    from PIL import Image
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor

    image_path = "data/examples/pc.png"
    image = Image.open(image_path)
    langs = ["en"]  # Replace with your languages or pass None (recommended to use None)
    recognition_predictor = RecognitionPredictor()
    detection_predictor = DetectionPredictor()

    predictions_bboxs = detection_predictor([image])
    predictions = recognition_predictor([image], [langs], detection_predictor)

    print(predictions)


def test_rapid_surya_ocr():
    import cv2
    from PIL import Image
    import os
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR
    from LonghorizonAgent.common import vis
    from surya.detection import DetectionPredictor

    image_path = "data/examples/web.png"
    image = Image.open(image_path)

    detection_predictor = DetectionPredictor()
    engine = RapidOCR()

    predict_bboxs = detection_predictor([image])

    ocr_bboxs = []
    for det_pred in predict_bboxs:
        ocr_bboxs.extend(np.array([p.polygon for p in det_pred.bboxes]))
    dt_boxes = engine.sorted_boxes(np.array(ocr_bboxs, dtype=np.float32))
    image_rgb = np.array(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
    img = engine.get_crop_img_list(image_bgr, dt_boxes)

    img, cls_res, cls_elapse = engine.text_cls(img)
    rec_res, rec_elapse = engine.text_rec(img, False)

    ocr_res = engine.get_final_res(
        dt_boxes, cls_res, rec_res, 0, cls_elapse, rec_elapse
    )
    result, elapse = ocr_res

    image_draw = vis.visualize_ocr_results(image_bgr, result)
    save_dir = f"./results/rapid_surya_ocr_tests"
    os.makedirs(save_dir, exist_ok=True)
    image_save_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(image_save_path, image_draw)
    print(image_save_path)
    pdb.set_trace()


def test_icon_detection():
    import cv2
    import os
    from LonghorizonAgent.perception.models.icon_detect_model import IconDetectModel
    from LonghorizonAgent.common import utils

    icon_detect_model = IconDetectModel()
    img_path = "tmp/screenshots/android_emulator-5554/b4cae513-e488-43a0-9a6b-fd76eb66486b.png"
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    icon_bboxs = icon_detect_model.detect(image_bgr,
                                          box_threshold=0.05,
                                          iou_threshold=0.1,
                                          imgsz=640,
                                          split_num_x=1,
                                          split_num_y=2,
                                          predict_type="combined")

    for bbox in icon_bboxs:
        x1, y1, x2, y2 = map(int, bbox[:4])  # 确保坐标是整数
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 5)
    result_dir = os.path.join("results", "icon-detect")
    os.makedirs(result_dir, exist_ok=True)

    img_save_path = os.path.join(result_dir, f"{os.path.basename(img_path)}")
    cv2.imwrite(img_save_path, image_bgr)
    print(img_save_path)


def test_icon_caption():
    import cv2
    import os
    from LonghorizonAgent.perception.models.icon_detect_model import IconDetectModel
    from LonghorizonAgent.perception.models.icon_caption_model import IconCaptionModel
    from LonghorizonAgent.common import utils

    icon_detect_model = IconDetectModel()
    icon_caption_model = IconCaptionModel()

    img_path = "data/examples/record/1742539226.7646315/screenshot/002.png"
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    icon_bboxs = icon_detect_model.detect(image_bgr,
                                          box_threshold=0.05,
                                          iou_threshold=0.1,
                                          imgsz=640,
                                          split_num_x=2,
                                          split_num_y=1,
                                          predict_type="combined")

    icon_texts = icon_caption_model.caption(image_rgb, icon_bboxs, prompt=None)

    for ii, bbox in enumerate(icon_bboxs):
        x1, y1, x2, y2 = map(int, bbox[:4])  # 确保坐标是整数
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色边界框，粗细为 2
        cv2.putText(image_bgr, f"{ii + 1}", ((x1 + x2) // 2, (y1 + y2) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 1)
        print(f"icon {ii + 1}: {icon_texts[ii]}")
    result_dir = os.path.join("results", "icon-caption")
    os.makedirs(result_dir, exist_ok=True)

    img_save_path = os.path.join(result_dir, os.path.basename(img_path))
    cv2.imwrite(img_save_path, image_bgr)
    print(img_save_path)


def test_ocr():
    import cv2
    import os
    from LonghorizonAgent.perception.models.ocr_model import OCRModel
    from LonghorizonAgent.common import utils

    ocr_model = OCRModel()
    img_path = "tmp/screenshots/android_emulator-5554/b4cae513-e488-43a0-9a6b-fd76eb66486b.png"
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ocr_results = ocr_model.ocr(image_bgr,
                                split_num_x=2,
                                split_num_y=1,
                                predict_type="single")

    for i, ocr_info in enumerate(ocr_results):
        bbox = ocr_info[0]
        x1, y1, x2, y2 = map(int, bbox[:4])  # 确保坐标是整数
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 5)
        print(i + 1, ocr_info[1])
    result_dir = os.path.join("results", "ocr-split")
    os.makedirs(result_dir, exist_ok=True)

    img_save_path = os.path.join(result_dir, f"{os.path.basename(img_path)}")
    cv2.imwrite(img_save_path, image_bgr)
    print(img_save_path)


def test_mobile_sam_model():
    """
    测试 Mobile SAM模型, 检查分割以及返回的box是否正确
    :return:
    """
    import cv2
    import os
    import numpy as np
    from LonghorizonAgent.perception.models.mobile_sam_model import MobileSAMModel

    sam_model = MobileSAMModel()

    image_path = "data/examples/record/task_1741230103.0275633/image/1.png"
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    image_mask, mask_box = sam_model.predict(image, point_coords=np.array([[w // 2, 3 * h // 4]]), point_labels=[1])
    assert len(mask_box) == 4
    result_dir = os.path.join("results", "mobile_sam-mask")
    os.makedirs(result_dir, exist_ok=True)
    img_save_path = os.path.join(result_dir, f"{os.path.basename(image_path)}")
    cv2.imwrite(img_save_path, image_mask)
    print(img_save_path)


def test_perception():
    import cv2
    import os
    from LonghorizonAgent.perception.screen_perception import ScreenPerception

    screen_p = ScreenPerception(use_icon_caption=False)

    img_path = "./tmp/agent_outputs/AutoExecutionAgent-6e1b90f1-9708-460a-8ebb-158b933fd457/screenshots/d668f0f5-7c19-44f4-8721-145f5e93a619.png"
    image_bgr = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    screen_status = screen_p.run_perception(image_bgr, split_x=2, split_y=2)

    for i, pinfo_ in enumerate(screen_status.perception_info):
        x1, y1, x2, y2 = pinfo_["box"]
        if pinfo_["type"] == "ocr_text":
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 5)
        elif pinfo_["type"] == "icon_detect":
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (255, 0, 0), 5)
        else:
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 5)
        print(i + 1, pinfo_["text"])

    result_dir = os.path.join("results", "perception")
    os.makedirs(result_dir, exist_ok=True)

    img_save_path = os.path.join(result_dir, f"{os.path.basename(img_path)}")
    cv2.imwrite(img_save_path, image_bgr)
    print(img_save_path)


if __name__ == '__main__':
    # test_rapid_ocr()
    # test_surya_ocr()
    # test_rapid_surya_ocr()
    # test_icon_detection()
    # test_icon_caption()
    # test_ocr()
    # test_mobile_sam_model()
    test_perception()
