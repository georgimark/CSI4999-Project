import cv2


def find_camera_indices():
    print("Scanning for cameras... (this may take a few seconds)")

    # scan the first 10 indexes
    for index in range(10):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            # Try to read a frame to ensure it works
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera found at Index {index}")
                # Optional: Display resolution
                w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                print(f"   - Resolution: {int(w)}x{int(h)}")
            else:
                print(f"⚠️  Camera found at Index {index}, but cannot read frame.")
            cap.release()

    print("Scan complete.")


if __name__ == "__main__":
    find_camera_indices()