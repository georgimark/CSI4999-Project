
#live_predict.py

#integrates live feed with predictions

import time
from ultralytics import YOLO
import cv2
from live_feed import open_camera


MODEL_PATH = ""
DEVICE = "0"
IMGSZ = 960
CONF = 0.25
USE_HALF = False
SHOW_FPS = True


def main():
    print(f"[INFO] loading YOLO11 model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    cap = open_camera()
    if cap is None or not cap.isOpened():
        print("[ERR] coulnt open camera.")
        return

    #just read whatever resolution the camera gives us
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] camera opened at {w}x{h}")

    cv2.namedWindow("YOLO live", cv2.WINDOW_NORMAL)

    last_time = time.time()
    frame_count = 0
    fps = 0.0

    #in progress