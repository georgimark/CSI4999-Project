import threading
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os
from datetime import datetime
import cv2
import time

from ultralytics import YOLO  # handles .pt / .onnx / .engine


class VideoController:
    def __init__(self, fps_var: tk.StringVar, conf_var: tk.IntVar, device: str = "0", imgsz: int = 1280):
        self.cap = None

        self.recording = False
        self.out = None

        self.model_path = None
        self.model = None  # YOLO model object

        self.running = False
        self.pause = False

        self.thread = None

        self.fps_var = fps_var
        self.conf_var = conf_var

        self.last_time = 0.0

        self.device = device
        self.imgsz = imgsz

        self.video_path = None  # <<< ADDED: stores selected video file path

    def open_camera(self):
        # Try AVFoundation sources first (macOS), then default 0
        if self.video_path is not None:  # <<< CHANGED: prefer file source if selected
            print(f"[INFO] Opening video file: {self.video_path}")
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                return cap
            print(f"[ERROR] Failed to open video file: {self.video_path}")
            cap.release()
            # fall through to camera if file fails

        for src in (0, 1):
            cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                return cap
            cap.release()

        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            return cap

        return None

    def start_video(self):
        if self.running:
            return

        print("[INFO] Starting video...")
        self.running = True
        self.pause = False
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

    def stop_video(self):
        print("[INFO] Stopping video...")
        self.running = False
        self.pause = False
        self.stop_recording()
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def toggle_pause(self):
        if not self.running:
            return
        self.pause = not self.pause
        print(f"[INFO] Pause set to {self.pause}")

    def start_recording(self):
        if not self.running or self.recording:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("videos", exist_ok=True)
        filename = f"videos/recording_{timestamp}.mp4"
        print(f"[INFO] Recording to {filename}...")

        self.out = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (1280, 720),
        )
        self.recording = True

    def stop_recording(self):
        if self.recording:
            print("[INFO] Stopped recording.")
            self.recording = False
            if self.out:
                self.out.release()
                self.out = None

    def select_model_file(self):
        filetypes = [("Model Files", "*.pt *.onnx *.engine"), ("All Files", "*.*")]
        path = filedialog.askopenfilename(title="Select YOLO Model", filetypes=filetypes)
        if path:
            self.model_path = path
            print(f"[INFO] Selected model: {path}")
            try:
                self.model = YOLO(self.model_path)  #  works for pt/onnx/engine
                print("[INFO] Model loaded successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
                self.model = None

    def select_video_file(self):  # <<< ADDED: choose a prerecorded video
        filetypes = [("Video Files", "*.mp4 *.mov *.avi *.mkv"), ("All Files", "*.*")]
        path = filedialog.askopenfilename(title="Select Video File", filetypes=filetypes)
        if path:
            self.video_path = path
            print(f"[INFO] Selected video file: {path}")
        else:
            print("[INFO] No video file selected.")

    def video_loop(self):
        self.cap = self.open_camera()
        if not self.cap:
            print("[ERROR] Unable to open camera.")
            self.running = False
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        cv2.namedWindow("Live Video", cv2.WINDOW_NORMAL)

        while self.running:
            if self.pause:
                cv2.waitKey(1)
                continue

            ok, frame = self.cap.read()
            if not ok:
                print("[WARN] Frame grab failed.")
                break

            start_time = time.time()
            conf = self.conf_var.get() / 100.0

            if self.model is not None:
                try:
                    results = self.model.predict(
                        source=frame,
                        device=self.device,
                        conf=conf,
                        imgsz=self.imgsz,
                        verbose=False,
                    )
                    annotated_frame = results[0].plot()
                except Exception as e:
                    print(f"[ERROR] Inference failed: {e}")
                    annotated_frame = frame
            else:
                annotated_frame = frame

            # FPS for full loop (including inference)
            dt = time.time() - start_time
            fps = 1.0 / dt if dt > 0 else 0.0
            self.fps_var.set(f"{fps:.2f}")

            if self.recording and self.out:
                self.out.write(annotated_frame)

            cv2.imshow("Live Video", annotated_frame)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                self.stop_video()
                break

        self.stop_video()


def main():
    root = tk.Tk()
    root.title("Control Panel")
    root.geometry("320x260")

    fps_var = tk.StringVar(value="0.00")
    conf_var = tk.IntVar(value=50)

    controller = VideoController(fps_var, conf_var, device="0", imgsz=1280)

    ttk.Label(root, text="Video Controls", font=("Arial", 14)).pack(pady=10)

    ttk.Button(root, text="Start Video", command=controller.start_video).pack(pady=5)
    ttk.Button(root, text="Stop Video", command=controller.stop_video).pack(pady=5)
    ttk.Button(root, text="Pause/Resume", command=controller.toggle_pause).pack(pady=5)
    ttk.Button(root, text="Start Recording", command=controller.start_recording).pack(pady=5)
    ttk.Button(root, text="Stop Recording", command=controller.stop_recording).pack(pady=5)
    ttk.Button(root, text="Select Model File", command=controller.select_model_file).pack(pady=5)
    ttk.Button(root, text="Select Video File", command=controller.select_video_file).pack(pady=5)  # <<< ADDED

    ttk.Label(root, text="Confidence Threshold (%)").pack(pady=(10, 0))
    ttk.Scale(root, from_=0, to=100, orient="horizontal", variable=conf_var).pack(pady=5)

    ttk.Label(root, text="FPS").pack(pady=(10, 0))
    ttk.Label(root, textvariable=fps_var, font=("Arial", 14)).pack()

    ttk.Button(root, text="Quit", command=root.quit).pack(pady=10)

    root.mainloop()
    controller.stop_video()


if __name__ == "__main__":
    main()
