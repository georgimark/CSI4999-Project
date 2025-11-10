# live_feed.py — realtime webcam preview at 1280x720 with clean shutdown (macOS friendly).

import cv2

def open_camera():
    # Try AVFoundation backend first (best on macOS), then default.
    for src in (0, 1):
        cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
        cap.release()
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return cap
    return None

def main():
    cap = open_camera()
    if cap is None:
        print("Could not open camera.")
        return

    # Force 1280x720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Show actual size
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened at {w}x{h}")

    cv2.namedWindow("Live", cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed.")
                break

            cv2.imshow("Live", frame)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):  # ESC or q
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
