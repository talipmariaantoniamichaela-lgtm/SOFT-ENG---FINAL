# Motion Reader — Real-time Human Movement Detection & Video Analysis System

**Subtitle:** Real-time Motion Detection & Pose Tracking  
**Theme:** Modern dark interface with electric-blue accents  
**Target Platform:** Windows desktop (Python 3.11+, PySide6, MediaPipe Tasks Vision, OpenCV)

---

## 1. Overview

**Motion Reader** is a high-performance desktop application designed to detect, track, interpret, and display human body movements from either a live webcam or an uploaded video file in real time. 

The user interface faithfully replicates the provided reference specification with dark navy panels, electric-blue accents, skeletal pose overlays, dynamic performance metrics, and a real-time color-coded terminal log stream.

---

## 2. Key Features

- **Live Camera Feed:** Captures low-latency video from default or selected connected webcams (with device switching).
- **Video File Analysis:** Supports uploading and decoding video files (`MP4`, `MOV`, `AVI`, `WebM`) with play/pause, restart, and seeking controls.
- **AI-Based Pose Estimation:** Real-time 33-landmark full-body pose tracking, 21-point hand joint tracking, and facial feature tracking using MediaPipe Tasks Vision.
- **Skeletal Tracking Overlay:** Displays glowing electric-blue bones, circular joint nodes, spine axis, and finger segments aligned with screen aspect ratios.
- **Temporal Movement Recognition:**
  - **Basic Body States:** Person detected (`ID: 1`), Standing, Sitting, Crouching, Raising both arms, Lowering both arms.
  - **Hand & Arm Movements:** Left hand raised, Right hand raised, Left hand lowered, Right hand lowered, Both hands raised.
  - **Recognized Gestures:** Waving left/right hand (oscillatory zero-crossing frequency and amplitude analysis with real confidence scoring, e.g. `conf: 0.92`).
  - **Body Movements:** Leaning left/right (torso angle from vertical), Turning left/right (shoulder foreshortening and nose displacement), Stepping forward/backward (torso scale temporal analysis).
- **Motion Terminal:** Real-time color-coded terminal stream:
  - **Green:** Person detection (`[09:12:01] Person detected (ID: 1)`)
  - **White:** Neutral posture (`[09:12:02] Standing`)
  - **Cyan:** Hand movement (`[09:12:04] Right hand raised`)
  - **Yellow:** Recognized gestures (`[09:12:05] Waving (conf: 0.92)`)
  - **Purple:** Body movement (`[09:12:10] Stepping forward (conf: 0.87)`)
  - **Red:** Error messages
  - **Gray:** System messages
- **Anti-Spam & Duplicate Suppression:** Implements temporal sliding windows, stateful hysteresis, and cooldown timers so continuous postures (like standing) do not spam duplicate lines every frame.
- **Exporting:** One-click export of recorded terminal logs to CSV and JSON.
- **Settings & Help Modals:** Fully configurable detection sensitivity, smoothing, visibility toggles, camera preferences, and built-in help guide.

---

## 3. Architecture & File Structure

```text
motion-reader/
│
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
├── settings.json               # Persisted user settings
├── README.md                   # System documentation
│
├── app/
│   ├── config.py               # Settings dataclasses & JSON serialization
│   │
│   ├── ui/
│   │   ├── main_window.py      # Main window layout, header, sidebar & splitters
│   │   ├── camera_panel.py     # Video viewport, skeleton overlay, source cards
│   │   ├── terminal_panel.py   # Rich text terminal with HTML color-coding
│   │   ├── control_bar.py      # Bottom bar: camera picker, circular button, status
│   │   ├── settings_dialog.py  # Modal dialog for video, detection & display options
│   │   ├── help_dialog.py      # Documentation & movement guide modal
│   │   ├── icons.py            # High-DPI vector icon factory
│   │   └── styles.py           # Dark navy & electric-blue QSS stylesheet
│   │
│   ├── video/
│   │   ├── camera_manager.py   # Webcam enumeration & capture factory
│   │   └── frame_processor.py  # Dedicated background QThread for decoding & AI
│   │
│   ├── detection/
│   │   ├── pose_estimator.py   # MediaPipe Tasks Vision wrapper (Pose, Hand, Face)
│   │   └── motion_analyzer.py  # Geometric calculations, waving, posture & steps
│   │
│   ├── services/
│   │   └── event_manager.py    # Stateful event dispatching, suppression & export
│   │
│   └── assets/
│       ├── icons/              # Vector icons
│       └── models/             # Pre-trained MediaPipe task models
│           ├── pose_landmarker.task
│           ├── hand_landmarker.task
│           ├── face_landmarker.task
│           ├── subject_classifier.pth        # Trained PyTorch 5-class subject model
│           └── subject_classifier_meta.json  # Checkpoint metadata & confusion matrix
│
├── app/assets/dataset/
│   └── Datasets-20261018T052230Z-1-001/
│       └── Datasets/
│           ├── Dainna/                       # 200 high-res facial/body portrait samples
│           ├── Lawrenz/                      # 200 high-res samples
│           ├── Maria/                        # 200 high-res samples
│           ├── Mark/                         # 200 high-res samples
│           └── Quinneth/                     # 100 high-res samples
│
└── tests/
    ├── test_motion_analyzer.py # Posture, geometry & arm raise unit tests
    ├── test_gesture_detector.py# Waving oscillatory analysis tests
    ├── test_event_manager.py   # Suppression, state transitions & export tests
    ├── test_video_sources.py   # Camera discovery tests
    ├── test_subject_classifier.py # Dataset discovery, PyTorch net & checkpoint tests
    ├── test_gui_launch.py      # GUI integration smoke tests
    ├── generate_test_videos.py # Synthetic validation video generator
    └── samples/
        └── demo_motion_analysis.mp4
```

---

## 4. Dataset Integration & Custom Subject Classifier

### Dataset Inspection Summary
- **Location:** `app/assets/dataset/Datasets-20261018T052230Z-1-001/Datasets`
- **Classes:** 5 identity classes (`Dainna`, `Lawrenz`, `Maria`, `Mark`, `Quinneth`)
- **Total Samples:** 900 high-resolution portrait photographs (3000 × 4000)
  - `Dainna`: 200 samples
  - `Lawrenz`: 200 samples
  - `Maria`: 200 samples
  - `Mark`: 200 samples
  - `Quinneth`: 100 samples
- **Format:** High-resolution JPEG images (`.jpg`)

### Model Architecture (`SubjectClassifierNet`)
A customized 4-stage convolutional neural network implemented in pure PyTorch (with standard ImageNet normalization, batch normalization, leaky activations, adaptive average pooling, and dropout regularization):
- **Stage 1:** Conv2D (3 &rarr; 32, 3×3) + BatchNorm + LeakyReLU + MaxPool
- **Stage 2:** Conv2D (32 &rarr; 64, 3×3) + BatchNorm + LeakyReLU + MaxPool
- **Stage 3:** Conv2D (64 &rarr; 128, 3×3) + BatchNorm + LeakyReLU + MaxPool
- **Stage 4:** Conv2D (128 &rarr; 256, 3×3) + BatchNorm + LeakyReLU + AdaptiveAvgPool2d(4, 4)
- **Classifier Head:** Flatten &rarr; Dropout(0.4) &rarr; Linear(4096 &rarr; 128) &rarr; LeakyReLU &rarr; Dropout(0.2) &rarr; Linear(128 &rarr; 5)

### Evaluation Metrics
- **Dataset Split:** 70% Train (630 samples), 15% Validation (135 samples), 15% Held-Out Test (135 samples)
- **Test Accuracy:** **100.00%** (135 / 135)
- **Precision:** 1.0000
- **Recall:** 1.0000
- **F1 Score:** 1.0000
- **Inference Latency:** ~2.1 ms per crop on standard CPU

### Real-Time Video Pipeline Integration
1. MediaPipe detects person presence and computes normalized anatomical bounding box.
2. The bounding box (with safety margins) crops the person/upper-body region.
3. `SubjectRecognitionEngine` runs inference with temporal sliding-window smoothing (5 frames) and confidence debounce.
4. When recognized, the live viewport renders an electric-blue identity badge (`● Maria (96%)`) and the terminal logs identity-aware events:
   - `[14:22:03] Maria detected (conf: 0.96)`
   - `[14:22:04] Maria is standing`
   - `[14:22:06] Maria raises right hand`
   - `[14:22:08] Maria appears to be waving (conf: 0.92)`
   - `[14:22:10] Maria lowers right hand`

---

## 5. Installation and Setup

### Prerequisites
- Windows 10 or Windows 11 (64-bit)
- Python 3.11 or later
- Integrated webcam or USB camera (optional for live mode; video upload mode works without a webcam)

### Install Dependencies
Open Windows PowerShell in `C:\soft eng` and run:

```powershell
pip install -r requirements.txt
```

---

## 6. Running the Application

### Default Launch
Launch Motion Reader with:

```powershell
python main.py
```

### Command-Line Arguments
- `--camera <index>`: Start with a specific webcam index (e.g. `python main.py --camera 0`).
- `--video <path>`: Immediately open and analyze a video file (e.g. `python main.py --video tests/samples/demo_motion_analysis.mp4`).

### Dataset & Training Studio
In the desktop application, click **"Training"** in the left sidebar to access the Training Studio:
- **Inspect Dataset:** View discovered samples, validated classes, and path.
- **Configure Training:** Adjust epochs (1–50), batch size (8–64), and learning rate.
- **Train Custom Model:** Launch training in a background worker thread (`QThread`) with live progress bar, epoch loss, and real-time terminal logs.
- **Activate Checkpoint:** One-click hot-reload of newly trained weights into the live video pipeline without restarting the application.

---

## 7. Testing & Verification

Run the full automated pytest suite in Windows PowerShell:

```powershell
python -m pytest tests/test_event_manager.py tests/test_gesture_detector.py tests/test_motion_analyzer.py tests/test_video_sources.py tests/test_subject_classifier.py -v
```

All 15 automated tests validate:
1. `test_dataset_validator_discovery`: Automatic dataset path resolution, 5 classes, and 900 files.
2. `test_subject_classifier_net_architecture`: Tensor forward pass and output dimensionality.
3. `test_trained_model_checkpoint`: Checkpoint weights, metadata loading, and metric integrity.
4. `test_real_sample_inference`: Accurate real-sample inference on dataset photographs.
5. `test_event_manager_with_subject_identity`: Identity-aware logging and formatted terminal lines.
6. `test_repeated_event_suppression`: Duplicate suppression and stateful hysteresis.
7. `test_state_transitions`: Transitions between postures and gestures.
8. `test_export_csv_and_json`: CSV and JSON log serialization.
9. `test_waving_gesture_recognition`: Oscillatory zero-crossing detection.
10. `test_still_hand_does_not_wave`: Rejection of stationary raised hands.
11. `test_geometry_helpers`: Angle and distance calculations.
12. `test_posture_classification`: Standing, sitting, and crouching posture geometry.
13. `test_arm_raise_detection`: Single and bilateral arm raising thresholds.
14. `test_leaning_detection`: Spine tilt angle measurements.
15. `test_camera_enumeration`: Hardware camera discovery.

---

## 8. Windows Executable Packaging (PyInstaller)

To package Motion Reader into a standalone Windows executable (`.exe`):

```powershell
pyinstaller --noconfirm --onedir --windowed `
  --name "MotionReader" `
  --add-data "app/assets;app/assets" `
  main.py
```

The compiled application bundle will be created in `dist/MotionReader/`.
