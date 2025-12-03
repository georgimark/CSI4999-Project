import threading
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os
from datetime import datetime
import cv2
import time
from PIL import Image, ImageTk

from ultralytics import YOLO


class VideoController:
    def __init__(self, root, fps_var, conf_var, status_var, imgsz_var):
        self.root = root
        self.fps_var = fps_var
        self.conf_var = conf_var
        self.status_var = status_var
        self.imgsz_var = imgsz_var

        self.cap = None
        self.recording = False
        self.out = None

        self.model_path = None
        self.model = None

        self.running = False
        self.pause = False
        self.thread = None

        self.device = "0"
        # Removed hardcoded self.imgsz = 640

        self.source_type = "camera"
        self.camera_index = 0
        self.video_file_path = ""

        self.video_label = tk.Label(root, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def set_source_camera(self, index_str):
        try:
            self.camera_index = int(index_str)
            self.source_type = "camera"
            self.status_var.set(f"Source set: Camera {self.camera_index}")
        except ValueError:
            self.status_var.set("Invalid Camera Index")

    def set_source_file(self):
        filetypes = [("Video Files", "*.mp4 *.mov *.avi *.mkv"), ("All Files", "*.*")]
        path = filedialog.askopenfilename(title="Select Video File", filetypes=filetypes)
        if path:
            self.video_file_path = path
            self.source_type = "file"
            self.status_var.set(f"Source set: {os.path.basename(path)}")

    def open_capture(self):
        if self.source_type == "camera":
            cap = cv2.VideoCapture(self.camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        else:
            if not os.path.exists(self.video_file_path):
                return None
            cap = cv2.VideoCapture(self.video_file_path)

        if cap.isOpened():
            return cap
        return None

    def start_video(self):
        if self.running:
            return

        self.cap = self.open_capture()
        if not self.cap:
            self.status_var.set("Error: Could not open source.")
            return

        print("[INFO] Starting video...")
        self.running = True
        self.pause = False
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()
        self.status_var.set("Running...")

    def stop_video(self):
        print("[INFO] Stopping video...")
        self.running = False
        self.pause = False
        self.stop_recording()

        time.sleep(0.2)

        if self.cap:
            self.cap.release()
            self.cap = None

        self.video_label.configure(image='')
        self.status_var.set("Stopped")

    def toggle_pause(self):
        if not self.running:
            return
        self.pause = not self.pause
        state = "Paused" if self.pause else "Running"
        self.status_var.set(state)

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
        self.status_var.set(f"Recording: {filename}")

    def stop_recording(self):
        if self.recording:
            print("[INFO] Stopped recording.")
            self.recording = False
            if self.out:
                self.out.release()
                self.out = None
            self.status_var.set("Recording saved.")

    def select_model_file(self):
        filetypes = [("Model Files", "*.pt *.onnx *.engine"), ("All Files", "*.*")]
        path = filedialog.askopenfilename(title="Select YOLO Model", filetypes=filetypes)
        if path:
            self.model_path = path
            print(f"[INFO] Selected model: {path}")
            try:
                self.model = YOLO(self.model_path)
                self.status_var.set(f"Loaded: {os.path.basename(path)}")
            except Exception as e:
                print(f"[ERROR] Failed to load model: {e}")
                self.status_var.set("Error loading model")
                self.model = None

    def video_loop(self):
        try:
            fps_in = self.cap.get(cv2.CAP_PROP_FPS)
            if fps_in <= 0: fps_in = 30
            delay = 1.0 / fps_in
        except:
            delay = 0.033

        while self.running:
            if self.pause:
                time.sleep(0.1)
                continue

            start_time = time.time()
            ok, frame = self.cap.read()

            if not ok:
                if self.source_type == "file":
                    print("[INFO] Video ended, looping...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print("[WARN] Camera disconnected.")
                    break

            conf = self.conf_var.get() / 100.0

            # GET CURRENT IMAGE SIZE FROM GUI
            try:
                current_imgsz = self.imgsz_var.get()
            except:
                current_imgsz = 1280  # Fallback safety

            annotated_frame = frame
            if self.model is not None:
                try:
                    results = self.model.predict(
                        source=frame,
                        device=self.device,
                        conf=conf,
                        imgsz=current_imgsz,
                        verbose=False,
                    )
                    annotated_frame = results[0].plot()
                except Exception as e:
                    print(f"[ERROR] Inference failed: {e}")

            disp_w, disp_h = 1280, 720
            if annotated_frame.shape[1] != disp_w or annotated_frame.shape[0] != disp_h:
                annotated_frame = cv2.resize(annotated_frame, (disp_w, disp_h))

            if self.recording and self.out:
                self.out.write(annotated_frame)

            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            dt = time.time() - start_time
            fps = 1.0 / dt if dt > 0 else 0.0
            self.fps_var.set(f"{fps:.2f}")

            if self.source_type == "file":
                proc_time = time.time() - start_time
                wait = delay - proc_time
                if wait > 0:
                    time.sleep(wait)

        self.running = False


def main():
    root = tk.Tk()
    root.title("Control Panel")
    root.geometry("1300x850")

    fps_var = tk.StringVar(value="0.00")
    conf_var = tk.IntVar(value=25)
    status_var = tk.StringVar(value="Ready")
    imgsz_var = tk.IntVar(value=1280)  # Default to 1280 to fix the error

    controller = VideoController(root, fps_var, conf_var, status_var, imgsz_var)

    control_frame = tk.Frame(root, bd=2, relief=tk.GROOVE)
    control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

    # 1. Source
    src_frame = tk.LabelFrame(control_frame, text="Input Source")
    src_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)

    cam_frame = tk.Frame(src_frame)
    cam_frame.pack(anchor="w")
    tk.Button(cam_frame, text="Set Camera Index:",
              command=lambda: controller.set_source_camera(cam_idx_entry.get())).pack(side=tk.LEFT)
    cam_idx_entry = tk.Entry(cam_frame, width=5)
    cam_idx_entry.insert(0, "0")
    cam_idx_entry.pack(side=tk.LEFT, padx=5)

    tk.Button(src_frame, text="Select Video File...", command=controller.set_source_file).pack(anchor="w", pady=2)

    # 2. Model
    mod_frame = tk.LabelFrame(control_frame, text="Model")
    mod_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
    tk.Button(mod_frame, text="Load .pt/.onnx", command=controller.select_model_file).pack(fill=tk.X)

    # NEW: Img Size Input
    sz_frame = tk.Frame(mod_frame)
    sz_frame.pack(pady=2)
    tk.Label(sz_frame, text="Img Size:").pack(side=tk.LEFT)
    tk.Entry(sz_frame, textvariable=imgsz_var, width=6).pack(side=tk.LEFT)

    tk.Label(mod_frame, text="Confidence %").pack()
    tk.Scale(mod_frame, from_=0, to=100, orient="horizontal", variable=conf_var).pack()

    # 3. Actions
    play_frame = tk.LabelFrame(control_frame, text="Actions")
    play_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
    tk.Button(play_frame, text="START Video", bg="#ddffdd", command=controller.start_video).pack(fill=tk.X)
    tk.Button(play_frame, text="STOP Video", bg="#ffdddd", command=controller.stop_video).pack(fill=tk.X)
    tk.Button(play_frame, text="Pause", command=controller.toggle_pause).pack(fill=tk.X)

    # 4. Info
    rec_frame = tk.LabelFrame(control_frame, text="Record / Info")
    rec_frame.pack(side=tk.LEFT, padx=5, fill=tk.Y)
    tk.Button(rec_frame, text="REC Start", command=controller.start_recording).pack(fill=tk.X)
    tk.Button(rec_frame, text="REC Stop", command=controller.stop_recording).pack(fill=tk.X)
    tk.Label(rec_frame, text="FPS:").pack(side=tk.LEFT)
    tk.Label(rec_frame, textvariable=fps_var, font=("Arial", 12, "bold")).pack(side=tk.LEFT)

    tk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

    root.mainloop()
    controller.stop_video()


if __name__ == "__main__":
    main()