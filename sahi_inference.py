"""
SAHI (Slicing Aided Hyper Inference) for VisDrone person detection.

SAHI slices each high-res image into overlapping patches, runs inference
per patch, then merges the detections — great for small object recall.
"""
import os
import glob
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def run_sahi_inference(
    model_path,
    images_dir,
    labels_dir,
    output_dir,
    slice_height=640,
    slice_width=640,
    overlap_ratio=0.2,
    conf=0.25,
    iou=0.45,
    device="cuda:0",
):
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        from sahi.utils.cv import read_image_as_pil
    except ImportError:
        print("SAHI not installed. Run: pip install sahi")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels_pred"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images_annotated"), exist_ok=True)

    # Load model via SAHI
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=conf,
        device=device,
    )

    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    print(f"Running SAHI inference on {len(image_files)} images...")
    print(f"  Slice: {slice_width}x{slice_height}  Overlap: {overlap_ratio}  Conf: {conf}  IoU: {iou}")

    # ---- Run inference and compute metrics ----
    tp_total = 0
    fp_total = 0
    fn_total = 0

    for img_path in tqdm(image_files, desc="SAHI"):
        filename = Path(img_path).name
        stem = Path(img_path).stem

        # SAHI sliced prediction
        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            postprocess_type="NMM",     # Non-Maximum Merging (better than NMS for sliced)
            postprocess_match_threshold=iou,
            verbose=0,
        )

        # Save predicted bboxes in YOLO format
        img = read_image_as_pil(img_path)
        w, h = img.size

        pred_lines = []
        predictions = result.object_prediction_list
        for pred in predictions:
            box = pred.bbox
            xc = ((box.minx + box.maxx) / 2.0) / w
            yc = ((box.miny + box.maxy) / 2.0) / h
            bw = (box.maxx - box.minx) / w
            bh = (box.maxy - box.miny) / h
            conf_val = pred.score.value
            pred_lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {conf_val:.4f}")

        pred_label_path = os.path.join(output_dir, "labels_pred", f"{stem}.txt")
        with open(pred_label_path, "w") as f:
            f.write("\n".join(pred_lines))

        # ---- Load GT and compute TP/FP/FN ----
        gt_label_path = os.path.join(labels_dir, f"{stem}.txt")
        gt_boxes = []
        if os.path.exists(gt_label_path):
            with open(gt_label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        gt_boxes.append([float(x) for x in parts[1:5]])

        pred_boxes = []
        for line in pred_lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                pred_boxes.append([float(x) for x in parts[1:5]])

        tp, fp, fn = compute_tp_fp_fn(pred_boxes, gt_boxes, iou_threshold=0.5)
        tp_total += tp
        fp_total += fp
        fn_total += fn

    # ---- Print final metrics ----
    precision = tp_total / (tp_total + fp_total + 1e-6)
    recall = tp_total / (tp_total + fn_total + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print("\n" + "="*50)
    print("SAHI Inference Results")
    print("="*50)
    print(f"  TP: {tp_total}  FP: {fp_total}  FN: {fn_total}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"\nPredictions saved to: {output_dir}/labels_pred/")


def box_iou(b1, b2):
    """Compute IoU between two YOLO-format boxes [xc, yc, w, h]."""
    def to_corners(b):
        x1 = b[0] - b[2] / 2
        y1 = b[1] - b[3] / 2
        x2 = b[0] + b[2] / 2
        y2 = b[1] + b[3] / 2
        return x1, y1, x2, y2

    ax1, ay1, ax2, ay2 = to_corners(b1)
    bx1, by1, bx2, by2 = to_corners(b2)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union_area = a_area + b_area - inter_area + 1e-6

    return inter_area / union_area


def compute_tp_fp_fn(pred_boxes, gt_boxes, iou_threshold=0.5):
    matched_gt = set()
    tp = 0
    fp = 0
    for pred in pred_boxes:
        best_iou = 0
        best_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_idx)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAHI inference on VisDrone val set")
    parser.add_argument("--model", default="/home/corsair/projects/visdrone/visdrone_runs/yolov8n_person_plus20/weights/best.pt")
    parser.add_argument("--images", default="/home/corsair/projects/visdrone/dataset_yolo/images/val")
    parser.add_argument("--labels", default="/home/corsair/projects/visdrone/dataset_yolo/labels/val")
    parser.add_argument("--output", default="/home/corsair/projects/visdrone/sahi_results")
    parser.add_argument("--slice-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    run_sahi_inference(
        model_path=args.model,
        images_dir=args.images,
        labels_dir=args.labels,
        output_dir=args.output,
        slice_height=args.slice_size,
        slice_width=args.slice_size,
        overlap_ratio=args.overlap,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )
