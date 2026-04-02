# VisDrone Advanced Tracking (YOLOv26s + SAHI + ORB + ECC + ByteTrack)

This repository contains an advanced, highly-optimized tracking pipeline specifically designed for the **VisDrone dataset**, which features extreme altitude drone footage, dense crowds, and sub-20 pixel pedestrian targets.

### Why This Pipeline Exists
Standard YOLO inference (`imgsz=640`) systematically fails to recall tiny pedestrians from 4k/1080p drone footage. Furthermore, standard tracking algorithms (like uncompensated ByteTrack) fragment IDs when the drone pans or tilts quickly.

This pipeline bridges state-of-the-art detection and classic computer vision techniques to achieve robust tracking where standard trackers fail.

---
# Detector
first we have used finetuned diffrent detectors to detect the objects in the frame and compared with baseline pretrained yolov8s(640x640 huggingface) . At the time of training we kept img resolution as 960px to detect small objects while at inference we took default 640px for fast inference.
            precision recall mAP50 mAP50-95
1. YOLOv26s .62         .58    .58    .25
2. YOLOv8s  .59         .57    .55    .22
3. yolov8n  .61         .58    .57    .22
4. huggingface 
yolov8s       .58        .38    .43    .20  


by the qualitative analysis we went through yolov26s for detection as it also have sahi implementation in it. 
## 1. Smart Tiling Detection (SAHI)
We implemented **Smart Tiling** using SAHI (Slicing Aided Hyper Inference) to dynamically adapt to the drone's altitude on a frame-by-frame basis:
- **Pass 1:** Fast, full-frame `1280px` YOLO inference.
- **Pass 2 (Fallback):** If fewer than 30 people are found, the pipeline assumes a high-altitude shot and triggers a heavy **2x2 grid slice** (`480p` tiles).
- **Pass 3 (Extreme):** If the shot is still practically empty, it triggers a **4x4 slice** (`270p` tiles) to capture microscopic targets.
- All detections are merged via global NMS (Non-Maximum Suppression).
- *Strict filter applied to isolate and track only the `person` class (Class 0), completely ignoring vehicles.*

## 2. Hybrid Motion Estimation (ORB + ECC)
To compensate for aggressive drone camera movements, we stabilize the coordinate space before tracking.
- **ORB Feature Matching:** Provides a fast, global homography initialization.
- **ECC Refinement:** Polishes the ORB initialization to sub-pixel accuracy.
- **Foreground Masking:** Moving pedestrians are explicitly masked out using YOLO bounding boxes before extracting Shi-Tomasi/ORB features, ensuring the tracker only aligns to the static background.

## 3. Track-Level Motion Compensation
Unlike naive implementations that warp detections (which desynchronizes bounding boxes from the current frame), this pipeline applies the estimated Homography matrix **directly to the Kalman Filter state space** inside ByteTrack. Tracking ID fragmentation is minimized as trajectories smoothly follow camera sweeps.

---


*Note: As tiling becomes more aggressive, Recall dramatically improves at the cost of bounding-box deduplication (Precision drop) and computational load (FPS drop). The Smart Tiling logic implemented in this repository perfectly balances this dynamically.*

---

## Usage

Run the tracking pipeline on any folder of sequential images (e.g., standard MOT17 format sequences):

```bash
conda run -n mambamot python orb+ecctracking.py \
  --img-dir dataset_mot/uav0000086_00000_v/img1 \
  --output output_tracking.mp4 \
  --model yolo26s.pt
```

The script will automatically overlay the live pipeline FPS speed onto the output video file. We have used rtx 5090 for all our experiment with VRAM 32 gb and recorded AVG fps of 9.67
