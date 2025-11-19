import threading
import tkinter as tk
from tkinter import ttk
import cv2
import time

class VideoController:
    def __init__(self):
        self.cap = None
        self.running = False
        self.thread = None

    
    def open_camera(self):
        for src in (0,1):
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
        self.thread = threading.Thread(target=self.video_loop, daemon=True)
        self.thread.start()

    def stop_video(self):
        print("[INFO] Stopping video...")
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

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
            ok, frame = self.cap.read()
            if not ok:
                print("[WARN] Frame grab failed.")
                break

            cv2.imshow("Live Video", frame)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                self.stop_video()
                break

            self.stop_video()

    def main():
        controller = VideoController()

        root = tk.Tk()
        root.title("Control Panel")
        root.geometry("300x200")

        ttk.Label(root, text="Video Controls", font=("Arial", 14)).pack(pady=10)

        ttk.Button(root, text="Start Video", command=controller.start_video).pack(pady=5)
        ttk.Button(root, text="Stop Video", command=controller.stop_video).pack(pady=5)
        ttk.Button(root, text="Quit", command=root.quit).pack(pady=20)

        root.mainloop()

        controller.stop_video()

    if __name__ == "__main__":
        main()