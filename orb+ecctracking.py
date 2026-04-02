import argparse
import cv2
import numpy as np
import sys
import os
import glob

sys.path.insert(0, os.path.abspath("ByteTrack"))
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# -------------------------------
# 🔹 ARGUMENTS & CONFIG
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--img-dir", type=str, default="dataset_mot/uav0000086_00000_v/img1")
parser.add_argument("--output",  type=str, default="output_tracking.mp4")
parser.add_argument("--model",   type=str, default="yolo26s.pt")
opts = parser.parse_args()

IMAGES_DIR  = opts.img_dir
MODEL_PATH  = opts.model
OUTPUT_PATH = opts.output

FRAME_RATE  = 30

# ── Detection thresholds ───────────────────────────────────────────────────────
CONF_THRESH       = 0.03   # Lowered from 0.05 — critical for tiny high-alt persons
SAHI_CONF_THRESH  = 0.02   # Even lower for SAHI tiles (NMS will clean up FPs)
SAHI_TRIGGER      = 30     # Run SAHI if full-frame finds fewer than this many persons
                           # Raised from 15 — at high altitude almost everything needs tiling

# ── SAHI tiling config ────────────────────────────────────────────────────────
# Key insight: at high altitude persons are ~8–20px tall.
# Smaller tiles = more zoom-in = better recall for tiny objects.
# 4x4 grid on 1920x1080 → 480x270 tiles, each person is relatively larger.
SAHI_SLICE_H      = 270    # Was 540 (2x2 grid) → now 270 (4x4 grid)
SAHI_SLICE_W      = 480    # Was 960              → now 480
SAHI_OVERLAP      = 0.3    # Slightly more overlap to avoid edge misses

# ── ORB config ────────────────────────────────────────────────────────────────
ORB_N_FEATURES    = 2000   # More features = more robust homography
ORB_SCALE         = 0.5    # Downscale for ORB matching (speed)

# ── ECC refinement config ─────────────────────────────────────────────────────
ECC_SCALE         = 0.5
ECC_MAX_ITER      = 30     # Fewer iters since ORB already gives a good init warp
ECC_EPS           = 1e-4
ECC_WARP_MODE     = cv2.MOTION_HOMOGRAPHY

# -------------------------------
# 🔹 INITIALIZE MODEL + TRACKER
# -------------------------------
model = YOLO(MODEL_PATH)

# Two SAHI models: aggressive (4x4) and conservative (2x2) tiling
sahi_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=MODEL_PATH,
    confidence_threshold=SAHI_CONF_THRESH,
    device="cuda:0"
)

class TrackerArgs:
    track_thresh  = 0.15   # Lowered slightly — accept weaker detections into tracks
    track_buffer  = 600    # 20s at 30fps — long buffer so high-alt tracks survive gaps
    match_thresh  = 0.8    # Slightly relaxed for small bbox IoU
    mot20         = True

tracker = BYTETracker(TrackerArgs(), frame_rate=FRAME_RATE)


# ═══════════════════════════════════════════════════════════════════════════════
# 🔹 ORB + ECC HYBRID MOTION ESTIMATION
#
# Strategy:
#   1. ORB feature matching gives a rough but fast homography H_orb
#   2. H_orb is used to pre-warp prev_gray into curr frame's coordinate space
#   3. ECC refines from this good initialisation (converges in far fewer iters)
#
# Why this beats plain ECC:
#   - ECC alone assumes small motion between frames (it's a local optimizer)
#   - Fast drone pans or altitude changes violate this assumption → ECC diverges
#   - ORB gives a global solution that handles large motions
#   - ECC then polishes sub-pixel accuracy that ORB can't achieve
# ═══════════════════════════════════════════════════════════════════════════════

orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES, scaleFactor=1.2, nlevels=8)
bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


def build_detection_mask(shape, detections, scale=1.0):
    """Black out detected object regions so features come from static background."""
    mask = np.ones(shape, dtype=np.uint8) * 255
    for det in detections:
        x1, y1, x2, y2 = (np.array(det[:4]) * scale).astype(int)
        # Expand box slightly — bounding boxes often clip the true object edge
        pad = 4
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(shape[1] - 1, x2 + pad), min(shape[0] - 1, y2 + pad)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
    return mask


def estimate_motion_orb(prev_gray, curr_gray, bg_mask=None):
    """
    Stage 1: ORB feature matching → rough homography H_orb.
    Returns H_orb (full resolution) and quality flag.
    """
    # Downscale for speed
    s_prev = cv2.resize(prev_gray, None, fx=ORB_SCALE, fy=ORB_SCALE, interpolation=cv2.INTER_AREA)
    s_curr = cv2.resize(curr_gray, None, fx=ORB_SCALE, fy=ORB_SCALE, interpolation=cv2.INTER_AREA)

    small_mask = None
    if bg_mask is not None:
        small_mask = cv2.resize(bg_mask, (s_curr.shape[1], s_curr.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

    kp1, des1 = orb.detectAndCompute(s_prev, small_mask)
    kp2, des2 = orb.detectAndCompute(s_curr, small_mask)

    if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
        return np.eye(3, dtype=np.float32), False

    matches = bf.match(des1, des2)
    # Keep only the best 50% of matches by Hamming distance
    matches = sorted(matches, key=lambda m: m.distance)
    matches = matches[:max(8, len(matches) // 2)]

    if len(matches) < 8:
        return np.eye(3, dtype=np.float32), False

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H_small, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H_small is None:
        return np.eye(3, dtype=np.float32), False

    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    if n_inliers < 6:
        return np.eye(3, dtype=np.float32), False

    # Scale translation back to full resolution
    H_small[0, 2] /= ORB_SCALE
    H_small[1, 2] /= ORB_SCALE

    return H_small.astype(np.float32), True


def estimate_motion_ecc_refine(prev_gray, curr_gray, H_init, bg_mask=None):
    """
    Stage 2: ECC refinement starting from H_init (the ORB homography).
    Pre-warps prev_gray using H_init so ECC only needs to correct residual error.
    Returns refined H (full resolution).
    """
    try:
        scale = ECC_SCALE
        h_full, w_full = curr_gray.shape

        # Pre-warp prev_gray into curr frame coords using ORB result
        # This makes the ECC problem small-motion, which is what it's good at
        prev_warped = cv2.warpPerspective(prev_gray, H_init, (w_full, h_full),
                                          flags=cv2.INTER_LINEAR)

        s_prev = cv2.resize(prev_warped, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        s_curr = cv2.resize(curr_gray,   None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        small_mask = None
        if bg_mask is not None:
            small_mask = cv2.resize(bg_mask, (s_curr.shape[1], s_curr.shape[0]),
                                    interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        # ECC now only needs to find the small residual warp H_residual
        warp_init = np.eye(3, 3, dtype=np.float32)
        criteria  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_MAX_ITER, ECC_EPS)

        _, H_residual_small = cv2.findTransformECC(
            s_prev, s_curr,
            warp_init,
            ECC_WARP_MODE,
            criteria,
            small_mask,
            5
        )

        # Scale residual translation
        H_residual_small[0, 2] /= scale
        H_residual_small[1, 2] /= scale

        # Final H = H_residual @ H_init  (apply ORB warp first, then residual)
        H_final = H_residual_small @ H_init
        return H_final.astype(np.float32)

    except cv2.error:
        # ECC failed — ORB result alone is still better than identity
        return H_init


def estimate_motion_hybrid(prev_gray, curr_gray, bg_mask=None):
    """
    Full ORB + ECC hybrid pipeline.
    Falls back gracefully at each stage.
    """
    H_orb, orb_ok = estimate_motion_orb(prev_gray, curr_gray, bg_mask)

    if not orb_ok:
        # No good ORB matches — try plain ECC from identity
        try:
            scale = ECC_SCALE
            s_prev = cv2.resize(prev_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            s_curr = cv2.resize(curr_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            warp   = np.eye(3, 3, dtype=np.float32)
            crit   = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_MAX_ITER, ECC_EPS)
            _, H_s = cv2.findTransformECC(s_prev, s_curr, warp, ECC_WARP_MODE, crit, None, 5)
            H_s[0, 2] /= scale
            H_s[1, 2] /= scale
            return H_s.astype(np.float32)
        except cv2.error:
            return np.eye(3, dtype=np.float32)

    # ORB succeeded — refine with ECC
    H_final = estimate_motion_ecc_refine(prev_gray, curr_gray, H_orb, bg_mask)
    return H_final


# ═══════════════════════════════════════════════════════════════════════════════
# 🔹 MOTION COMPENSATION ON TRACKER STATE
# ═══════════════════════════════════════════════════════════════════════════════

def compensate_tracks(tracker, H):
    for strack in tracker.tracked_stracks + tracker.lost_stracks:
        mean = strack.mean
        if mean is None:
            continue

        pts = np.array([[[mean[0], mean[1]]]], dtype=np.float32)
        pts_w = cv2.perspectiveTransform(pts, H)
        new_cx, new_cy = pts_w[0][0]

        vel_pts = np.array([[[mean[0] + mean[4], mean[1] + mean[5]]]], dtype=np.float32)
        vel_w = cv2.perspectiveTransform(vel_pts, H)
        new_vx = vel_w[0][0][0] - new_cx
        new_vy = vel_w[0][0][1] - new_cy

        strack.mean[0] = new_cx
        strack.mean[1] = new_cy
        strack.mean[4] = new_vx
        strack.mean[5] = new_vy


# ═══════════════════════════════════════════════════════════════════════════════
# 🔹 DETECTION — MULTI-SCALE SAHI
#
# Key fix for high-altitude missing persons:
#   - 4x4 tiling (480x270) instead of 2x2 (960x540)
#   - Lower SAHI conf threshold (0.02)
#   - Two-pass: run 2x2 first, if still sparse run 4x4 on top
#   - NMS at the end to clean up duplicates from overlapping tiles
# ═══════════════════════════════════════════════════════════════════════════════

def run_nms(detections, iou_thresh=0.5):
    """Standard NMS to clean duplicate detections from overlapping SAHI tiles."""
    if len(detections) == 0:
        return detections
    boxes  = np.array([[d[0], d[1], d[2], d[3]] for d in detections], dtype=np.float32)
    scores = np.array([d[4] for d in detections], dtype=np.float32)
    # cv2.dnn.NMSBoxes expects (x,y,w,h)
    boxes_xywh = boxes.copy()
    boxes_xywh[:, 2] -= boxes_xywh[:, 0]
    boxes_xywh[:, 3] -= boxes_xywh[:, 1]
    indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), SAHI_CONF_THRESH, iou_thresh)
    if len(indices) == 0:
        return []
    indices = indices.flatten()
    return [detections[i] for i in indices]


def detect_persons(frame, model, sahi_model):
    """
    Multi-scale detection optimised for high-altitude tiny persons.

    Pass 1: Full-frame YOLO (fast, catches medium/large persons)
    Pass 2: 2x2 SAHI (catches persons missed by full-frame)
    Pass 3: 4x4 SAHI (only if still too few — aggressive, catches sub-20px persons)
    Final:  NMS across all passes
    """
    detections = []

    # ── Pass 1: Full frame ────────────────────────────────────────────────────
    results = model(frame, conf=CONF_THRESH, imgsz=1280, verbose=False)[0]
    if results.boxes is not None:
        for r in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = r
            if int(cls) == 0:
                detections.append([x1, y1, x2, y2, conf])

    # ── Pass 2: 2x2 SAHI (480p tiles) ────────────────────────────────────────
    if len(detections) < SAHI_TRIGGER:
        result_2x2 = get_sliced_prediction(
            frame, sahi_model,
            slice_height=540, slice_width=960,
            overlap_height_ratio=SAHI_OVERLAP,
            overlap_width_ratio=SAHI_OVERLAP,
            perform_standard_pred=False,
            verbose=0
        )
        for obj in result_2x2.object_prediction_list:
            if obj.category.id == 0:
                b = obj.bbox.to_xyxy()
                detections.append([b[0], b[1], b[2], b[3], obj.score.value])

    # ── Pass 3: 4x4 SAHI (270p tiles) — high-altitude rescue ─────────────────
    # Only fires when scene is very sparse (drone very high up)
    if len(detections) < SAHI_TRIGGER // 2:
        result_4x4 = get_sliced_prediction(
            frame, sahi_model,
            slice_height=SAHI_SLICE_H,
            slice_width=SAHI_SLICE_W,
            overlap_height_ratio=SAHI_OVERLAP,
            overlap_width_ratio=SAHI_OVERLAP,
            perform_standard_pred=False,
            verbose=0
        )
        for obj in result_4x4.object_prediction_list:
            if obj.category.id == 0:
                b = obj.bbox.to_xyxy()
                detections.append([b[0], b[1], b[2], b[3], obj.score.value])

    # ── NMS across all passes ─────────────────────────────────────────────────
    detections = run_nms(detections, iou_thresh=0.5)
    return detections


# ═══════════════════════════════════════════════════════════════════════════════
# 🔹 VIDEO SETUP
# ═══════════════════════════════════════════════════════════════════════════════

image_files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
if not image_files:
    print(f"Error: No images found in {IMAGES_DIR}")
    sys.exit(1)

first_frame = cv2.imread(image_files[0])
height, width = first_frame.shape[:2]

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    FRAME_RATE,
    (width, height)
)

prev_gray       = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
track_history   = {}
frame_id        = 0
prev_detections = []

import time
total_time = 0.0
processed_frames = 0

# ═══════════════════════════════════════════════════════════════════════════════
# 🔹 MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

for img_path in image_files[1:]:
    loop_start = time.time()
    frame = cv2.imread(img_path)
    if frame is None:
        continue

    frame_id += 1
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Step 1: Build bg mask from previous detections ────────────────────────
    bg_mask = build_detection_mask(curr_gray.shape, prev_detections)

    # ── Step 2: ORB + ECC hybrid motion estimation ────────────────────────────
    H = estimate_motion_hybrid(prev_gray, curr_gray, bg_mask=bg_mask)
    prev_gray = curr_gray.copy()

    # ── Step 3: Multi-scale detection ─────────────────────────────────────────
    detections = detect_persons(frame, model, sahi_model)
    prev_detections = detections  # Save for next frame's bg mask

    detections_np = np.array(detections) if len(detections) > 0 else np.empty((0, 5))

    # ── Step 4: Apply motion compensation to Kalman states ───────────────────
    compensate_tracks(tracker, H)

    # ── Step 5: ByteTrack update ──────────────────────────────────────────────
    online_targets = tracker.update(detections_np, [height, width], [height, width])

    # ── Step 6: Draw results ──────────────────────────────────────────────────
    for t in online_targets:
        tlwh = t.tlwh
        tid  = t.track_id
        x, y, w, h = map(int, tlwh)
        cx, cy = int(x + w / 2), int(y + h / 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tid}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if tid not in track_history:
            track_history[tid] = []
        track_history[tid].append((cx, cy))
        if len(track_history[tid]) > 30:
            track_history[tid].pop(0)

        for i in range(1, len(track_history[tid])):
            cv2.line(frame, track_history[tid][i - 1], track_history[tid][i], (255, 0, 0), 1)

    loop_end = time.time()
    frame_time = loop_end - loop_start
    total_time += frame_time
    processed_frames += 1
    fps = 1.0 / (frame_time + 1e-9)

    print(f"[Frame {frame_id:04d}] Detections: {len(detections)} | Time: {frame_time:.3f}s | FPS: {fps:.1f}", flush=True)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    out.write(frame)

# ═══════════════════════════════════════════════════════════════════════════════
# 🔹 CLEANUP
# ═══════════════════════════════════════════════════════════════════════════════
out.release()
cv2.destroyAllWindows()
avg_fps = processed_frames / total_time if total_time > 0 else 0
print(f"\nDone! Saved to {OUTPUT_PATH}")
print(f"Average Pipeline FPS: {avg_fps:.2f}")