# RoadVision

## Contributors (username and real name)
- **georgimark** (Mark Georgi)  
- **PatRobin02** (Patrick Robin)  
- **Alexandraivezaj** (Alexandra Ivezaj)  
- **wmkumpula** (William Kumpula)  
- **Rita** (Rita Mansoor)

RoadVision is a computer vision system designed for real-time and prerecorded object detection on roadways using **YOLOv11**. The system includes a graphical user interface for live inference control, tools for batch video processing, and a complete training pipeline.

---

## Directory Structure

```text
CSI4999-Project/
│
├── runs/                      # Training artifacts
│   └── detect/
│       └── bdd11s_base/
│           └── best.onnx      # Trained model weights
├── live_feed_test/
│   └── live_feed.py               # Camera backend logic
│   └──live_predict.py            # Standalone real-time inference script
├── src/
│   └── predict_video.py           # Batch video processing script
├── tools/
│   └── train_bdd.py               # YOLOv11 training pipeline
│   └──verify_dataset.py          # Dataset consistency checker
│   └── find_camera_id.py          # Utility to scan for available camera indices
├── UI/
│   └── control_panel.py       # Main Graphical User Interface (Full integrated System)
├── videos/                    # Output directory for GUI recordings
├── outputs/                   # Output directory for batch processed videos
├── README.md
└── requirements.txt           # Dependencies file
```

---

## Setup & Installation

### Install Requirements
Ensure you have Python installed, then install the necessary dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage Instructions

### 1. Launch the Application
Run the Control Panel script from the project root:

```bash
python UI/control_panel.py
```

### 2. Load the Model
Once the **Control Panel** window opens:

- Locate the **Model** section.  
- Click **Load .pt/.onnx**.  
- Navigate to the weights file (e.g., `runs/detect/bdd11s_base/best.onnx`) and select it.

---

### 3. Select Input Source

#### For Live Camera:
- Enter the camera index (usually `0` or `1`) in the text box.
- Click **Set Camera Index**.

**Tip:** If you don't know your camera index, run:

```bash
python tools/find_camera_id.py
```

#### For Video File:
- Click **Select Video File...** and choose a supported video file (`.mp4`, `.mov`, `.avi`, `.mkv`).

---

### 4. Start Inference
Click the green **START Video** button to begin the feed and object detection.

---

### 5. Adjust Settings
- **Confidence:** Use the slider to adjust the detection confidence threshold (0–100%) in real time.  
- **Image Size:** Adjust inference image size (default `1280`) in the **Img Size** box.  

---

### 6. Recording
To save a video of the annotated feed:

- Click **REC Start** to begin recording.  
- Click **REC Stop** to finish.

**Note:** Recorded videos will be saved in the `videos/` directory with a timestamped filename (e.g., `recording_20231025_120000.mp4`).

---

### 7. Stop Application
Click **STOP Video** to end the feed.

