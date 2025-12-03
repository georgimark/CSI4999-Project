
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
    # added basic check + config print
    if not MODEL_PATH:
        print("[ERR] MODEL_PATH is empty. Please set MODEL_PATH at the top of live_predict.py.")
        return

    print(f"[INFO] loading YOLO11 model: {MODEL_PATH}")
    print(f"[INFO] settings -> device={DEVICE}, imgsz={IMGSZ}, conf={CONF}")
    model = YOLO(MODEL_PATH)


    cap = open_camera()
    if cap is None or not cap.isOpened():
        print("[ERR] coulnt open camera.")
        return

    # just read whatever resolution the camera gives us
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

            # yolo prediction for this frame
            results = model.predict(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF,
                device=DEVICE,
                half=USE_HALF,
                verbose=False,
            )

            annotated = results[0].plot()

            # if yolo changed size, resize back to camera size
            if annotated.shape[1] != w or annotated.shape[0] != h:
                annotated = cv2.resize(
                    annotated, (w, h), interpolation=cv2.INTER_LINEAR
                )

            # fps counter
            frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                fps = frame_count / (now - last_time)
                frame_count = 0
                last_time = now

            if SHOW_FPS:
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("YOLO Live", annotated)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):  #ESC or q
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Live session ended.")


if __name__ == "__main__":
    main()
                
