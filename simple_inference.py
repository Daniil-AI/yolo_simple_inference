from typing import List, Tuple, Dict
import cv2
import numpy as np
from ultralytics import YOLO
from utils.encryption import decrypt_model
import os
import warnings
import torch

# Ключ для расшифровки весов (пример)
SECRET_STRING = "9SlEUBpmy74_LmOcOUE9BAiaGQN2BzB2Pwc-Jet6hkc="

# Список имен классов НЕ МЕНЯТЬ
CLASS_NAMES: List[str] = [
    "Truck",
    "Car",
]

# Цвета отображения ббоксов для каждого из классов
COLORS: Dict[int, tuple] = {
    0: (0, 0, 128),
    1: (0, 128, 0),
}

warnings.filterwarnings("ignore")


# Функция для выполнения инференса на одном кадре
def perform_inference(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    imgsz: Tuple[int, int],
) -> List[dict]:
    results = model.predict(frame, conf=conf, imgsz=imgsz, batch=1, verbose=False)
    detections: List[dict] = []

    for result in results:
        for box in result.boxes:
            bbox = box.xyxy[0]
            detection: dict = {
                "class_id": box.cls,
                "confidence": box.conf,
                "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
            }
            detections.append(detection)
    return detections


# Функция для отображения рамок и текста
def draw_rectangle(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    title: str,
    opacity: float = 0.15,
) -> np.ndarray:
    x1, y1 = pt1
    x2, y2 = pt2
    scene = img.copy()
    # Top left
    cv2.rectangle(
        img=img,
        pt1=(x1, y1),
        pt2=(x2, y2),
        color=color,
        thickness=-1,
    )
    img = cv2.addWeighted(img, opacity, scene, 1 - opacity, gamma=0)

    cv2.rectangle(img, (x1, y1 - 20), (x1 + len(title) * 14, y1), color, -1)
    cv2.putText(
        img, title, (x1, y1 - 2), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2
    )
    return img


# Функция для отображения результатов инференса на кадре
def draw_detections(frame: np.ndarray, detections: List[dict]) -> np.ndarray:
    for detection in detections:
        class_id: int = int(detection["class_id"])
        class_name: str = CLASS_NAMES[class_id]
        confidence: float = float(detection["confidence"])
        bbox: Tuple[int, int, int, int] = tuple(map(int, detection["bbox"]))
        x1, y1, x2, y2 = bbox
        title = f"{class_id} {class_name} {round(confidence, 2)}"
        frame = draw_rectangle(
            img=frame, pt1=(x1, y1), pt2=(x2, y2), color=COLORS[class_id], title=title
        )
    return frame


# Основная функция для обработки видео
def process_video(
    video_path: str,
    model: YOLO,
    conf: float,
    imgsz: Tuple[int, int],
) -> None:
    cap: cv2.VideoCapture = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections: List[dict] = perform_inference(model, frame, conf, imgsz)
        result_frame: np.ndarray = draw_detections(frame, detections)
        cv2.imshow("Video Detections", result_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


# Основная функция для обработки видео и его последуюшей записи
def process_and_draw_video(
    video_path: str,
    end_video_path: str,
    model: YOLO,
    conf: float,
    imgsz: Tuple[int, int],
) -> None:
    cap: cv2.VideoCapture = cv2.VideoCapture(video_path)

    # создание обьекта видео
    fourcc: cv2.VideoWriter_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps: int = int(cap.get(cv2.CAP_PROP_FPS))
    w: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out: cv2.VideoCapture = cv2.VideoWriter(end_video_path, fourcc, fps, (w, h))

    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections: List[dict] = perform_inference(model, frame, conf, imgsz)
        result_frame: np.ndarray = draw_detections(frame, detections)

        out.write(result_frame)
        cv2.imshow("Video Detections", result_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()


# Поиск видео и картинок в папке
def search_input_files(data_dir: str) -> [List[str], List[str]]:
    video_list: List[str] = []
    image_list: List[str] = []
    for file in os.listdir(data_dir):
        if file.lower().endswith(("mov", "mp4", "avi")):
            video_list.append(file)
        elif file.lower().endswith(("jpg", "png")):
            image_list.append(file)
    return video_list, image_list


# Заспуск единичной деткции
def detect_video(
    draw_video: bool,
    model: YOLO,
    start_video_path: str,
    end_video_path: str,
    conf: float,
    imgsz: Tuple[int, int],
) -> None:
    if draw_video:
        process_and_draw_video(start_video_path, end_video_path, model, conf, imgsz)
    else:
        process_video(start_video_path, model, conf, imgsz)


# Функция для обработки изображения
def detect_image(
    image_path: str,
    end_image_path: str,
    model: YOLO,
    conf: float,
    imgsz: Tuple[int, int],
) -> None:
    image: np.ndarray = cv2.imread(image_path)
    detections: List[dict] = perform_inference(model, image, conf, imgsz)
    result_image: np.ndarray = draw_detections(image, detections)

    cv2.imwrite(end_image_path, result_image)


def run_simple_inference(
    model_path: str,
    dummy_model_path: str,
    start_data_dir: str,
    end_data_dir: str,
    conf: float,
    imgsz: Tuple[int, int],
    encrypted_weight: bool,
    draw_video: bool,
) -> None:

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if encrypted_weight:
        model = decrypt_model(model_path, dummy_model_path, SECRET_STRING, device)
        print("Weights decrypted")
    else:
        model = YOLO(model_path)

    if not os.path.exists(end_data_dir):
        os.makedirs(end_data_dir)

    # Основной цикл программы
    video_list, image_list = search_input_files(start_data_dir)
    video_num: int = len(video_list)
    image_num: int = len(image_list)

    for i, image in enumerate(image_list):
        file_path = os.path.join(start_data_dir, image)
        result_file_path = os.path.join(end_data_dir, image)
        detect_image(
            file_path,
            result_file_path,
            model,
            conf,
            imgsz,
        )
        if (i + 1) % 10 == 0:
            print(f"Image {i+1}/{image_num} processed")
    if image_num:
        print(f"All images processed")

    for i, video in enumerate(video_list):
        file_path = os.path.join(start_data_dir, video)
        result_file_path = os.path.join(end_data_dir, video)
        detect_video(
            draw_video,
            model,
            file_path,
            result_file_path,
            conf,
            imgsz,
        )
        print(f"Video {i+1}/{video_num} processed")
    if video_num:
        print("All videos processed")
