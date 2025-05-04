import os
import face_recognition
from tqdm import tqdm
from dotenv import load_dotenv
import mediapipe as mp
from ultralytics import YOLO
import cv2

def has_face_yolo(path, model):
    img = cv2.imread(path)
    if img is None:
        return False
    results = model(img, verbose=False)
    return len(results[0].boxes) > 0

def has_face_mediapipe(path, face_detector):
    img = cv2.imread(path)
    if img is None:
        return False
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_detector.process(img_rgb)
    return result.detections is not None and len(result.detections) > 0

def has_face_recognition(path):
    try:
        img = face_recognition.load_image_file(path)
        faces = face_recognition.face_locations(img)
        return len(faces) > 0
    except Exception:
        return True  # por si falla, asumimos que puede tener algo

if __name__ == "__main__":
    load_dotenv("world/.env")
    REAL_IMG_PATH = os.getenv("real_img_path")
    # REAL_IMG_PATH = os.getenv("open_images_path")
    input_dir = REAL_IMG_PATH
    output_face_dir = "data/faces_media_pipe/faces/"
    output_no_face_dir = "data/faces_media_pipe/no_faces/"

    os.makedirs(output_face_dir, exist_ok=True)
    os.makedirs(output_no_face_dir, exist_ok=True)

    face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
    # model = YOLO("yolov8n-face.pt")  # Usa yolov8s-face.pt si quieres más precisión

    files = [f for f in os.listdir(input_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]



    mediapipe_face_count = 0
    mediapipe_no_face_count = 0
    yolo_face_count = 0

    face_recognition_face_count = 0
    face_recognition_no_face_count = 0
    yolo_no_face_count = 0

    for fname in tqdm(files):
        in_path = os.path.join(input_dir, fname)
        if has_face_mediapipe(in_path, face_detector):
            mediapipe_face_count += 1
        else:
            mediapipe_no_face_count += 1

        if has_face_recognition(in_path):
            face_recognition_face_count += 1
        else:
            face_recognition_no_face_count += 1

        # if has_face_yolo(in_path, model):
        #     yolo_face_count += 1
        # else:
        #     yolo_no_face_count += 1

    total_mediapipe = mediapipe_face_count + mediapipe_no_face_count
    porcentaje_mediapipe = (mediapipe_face_count / total_mediapipe) * 100 if total_mediapipe > 0 else 0
    print(f"✅ Caras detectadas: {mediapipe_face_count} / {total_mediapipe} ({porcentaje_mediapipe:.2f}%)")

    total_face_recognition = face_recognition_face_count + face_recognition_no_face_count
    porcentaje_face_recognition = (face_recognition_face_count / total_face_recognition) * 100 if total_face_recognition > 0 else 0
    print(f"✅ Caras detectadas: {face_recognition_face_count} / {total_face_recognition} ({porcentaje_face_recognition:.2f}%)")

    # total_yolo = yolo_face_count + yolo_no_face_count
    # porcentaje_yolo = (yolo_face_count / total_yolo) * 100 if total_yolo > 0 else 0
    # print(f"✅ Caras detectadas: {yolo_face_count} / {total_yolo} ({porcentaje_yolo:.2f}%)")
