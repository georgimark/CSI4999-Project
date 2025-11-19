
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

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] frame grab failed.")
                break

            #yolo prediction for this frame
            results = model.predict(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF,
                device=DEVICE,
                half=USE_HALF,
                verbose=False,
            )

            annotated = results[0].plot()

            #if yolo changed size, resize back to camera size
            if annotated.shape[1] != w or annotated.shape[0] != h:
                annotated = cv2.resize(
                    annotated, (w, h), interpolation=cv2.INTER_LINEAR
                )

                
if __name__ == "__main__":
    main()
