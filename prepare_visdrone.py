import os
import cv2
import shutil
import glob

def convert_det_to_yolo(visdrone_dir, output_dir):
    """
    Converts VisDrone2019-DET-train to YOLO format mapping pedestrian (1) and person (2) to class 0.
    """
    print(f"Converting DET dataset from {visdrone_dir} to {output_dir} in YOLO format...")
    
    images_dir = os.path.join(visdrone_dir, 'images')
    ann_dir = os.path.join(visdrone_dir, 'annotations')
    
    out_images_dir = os.path.join(output_dir, 'images', 'train')
    out_labels_dir = os.path.join(output_dir, 'labels', 'train')
    
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    
    # Valid classes in VisDrone: 1: pedestrian, 2: person
    valid_classes = {1: 0, 2: 0} # map to 0
    
    image_files = glob.glob(os.path.join(images_dir, '*.jpg'))
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)
        ann_path = os.path.join(ann_dir, f"{name}.txt")
        
        if not os.path.exists(ann_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w, _ = img.shape
        
        out_label_path = os.path.join(out_labels_dir, f"{name}.txt")
        yolo_lines = []
        
        with open(ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                    
                bbox_left = int(parts[0])
                bbox_top = int(parts[1])
                bbox_width = int(parts[2])
                bbox_height = int(parts[3])
                score = int(parts[4])
                category = int(parts[5])
                
                # Check category
                if category not in valid_classes:
                    continue
                    
                # YOLO format: class x_center y_center width height (normalized)
                x_center = (bbox_left + bbox_width / 2.0) / w
                y_center = (bbox_top + bbox_height / 2.0) / h
                width = bbox_width / w
                height = bbox_height / h
                
                # Clip between 0 and 1
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width = max(0.0, min(1.0, width))
                height = max(0.0, min(1.0, height))

                yolo_class = valid_classes[category]
                yolo_lines.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
        if yolo_lines:
            # Copy image only if it has valid annotations (optional, but good practice)
            shutil.copy(img_path, os.path.join(out_images_dir, filename))
            with open(out_label_path, 'w') as f:
                f.writelines(yolo_lines)
                
    print(f"Finished converting DET dataset.")

def convert_mot_to_yolo(visdrone_dir, output_dir):
    """
    Converts VisDrone2019-MOT-val to YOLO detection format for validation.
    """
    print(f"Converting MOT dataset from {visdrone_dir} to {output_dir} in YOLO format for validation...")
    
    seqs_dir = os.path.join(visdrone_dir, 'sequences')
    ann_dir = os.path.join(visdrone_dir, 'annotations')
    
    out_images_dir = os.path.join(output_dir, 'images', 'val')
    out_labels_dir = os.path.join(output_dir, 'labels', 'val')
    
    os.makedirs(out_images_dir, exist_ok=True)
    os.makedirs(out_labels_dir, exist_ok=True)
    
    valid_classes = {1: 0, 2: 0} # map to 0
    
    seq_dirs = [d for d in os.listdir(seqs_dir) if os.path.isdir(os.path.join(seqs_dir, d))]
    
    for seq in seq_dirs:
        seq_path = os.path.join(seqs_dir, seq)
        ann_path = os.path.join(ann_dir, f"{seq}.txt")
        
        if not os.path.exists(ann_path):
            continue
            
        # Get sequence images
        image_files = sorted(glob.glob(os.path.join(seq_path, '*.jpg')))
        if not image_files:
            continue
            
        # Get image dimensions from first image
        sample_img = cv2.imread(image_files[0])
        if sample_img is None:
            continue
        h, w, _ = sample_img.shape
        
        # Parse annotations and group by frame
        frame_annotations = {}
        with open(ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                frame_idx = int(parts[0])
                category = int(parts[7])
                
                if category not in valid_classes:
                    continue
                    
                bbox_left = int(parts[2])
                bbox_top = int(parts[3])
                bbox_width = int(parts[4])
                bbox_height = int(parts[5])
                
                # YOLO format (normalized)
                x_center = (bbox_left + bbox_width / 2.0) / w
                y_center = (bbox_top + bbox_height / 2.0) / h
                width = bbox_width / w
                height = bbox_height / h
                
                # Clip
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width = max(0.0, min(1.0, width))
                height = max(0.0, min(1.0, height))
                
                yolo_class = valid_classes[category]
                yolo_line = f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                
                if frame_idx not in frame_annotations:
                    frame_annotations[frame_idx] = []
                frame_annotations[frame_idx].append(yolo_line)
        
        # Copy images and write labels
        for img_path in image_files:
            filename = os.path.basename(img_path)
            # Typically frame names are numbers, let's prefix with seq name to avoid collision
            name, ext = os.path.splitext(filename)
            # In VisDrone sequences, images are 0000001.jpg etc.
            # but annotations use frame_idx which matches the numeric filename.
            frame_idx = int(name) 
            
            if frame_idx in frame_annotations:
                new_filename = f"{seq}_{filename}"
                shutil.copy(img_path, os.path.join(out_images_dir, new_filename))
                
                out_label_path = os.path.join(out_labels_dir, f"{seq}_{name}.txt")
                with open(out_label_path, 'w') as f:
                    f.writelines(frame_annotations[frame_idx])
                    
    print(f"Finished converting MOT dataset to YOLO format.")

if __name__ == "__main__":
    base_dir = "/home/corsair/projects/visdrone"
    
    det_train = os.path.join(base_dir, "VisDrone2019-DET-train")
    mot_val = os.path.join(base_dir, "VisDrone2019-MOT-val")
    
    yolo_out = os.path.join(base_dir, "dataset_yolo")
    mot_out = os.path.join(base_dir, "dataset_mot")
    
    # 1. Convert DET Train (Detection format)
    if os.path.exists(det_train):
        convert_det_to_yolo(det_train, yolo_out)
    else:
        print(f"DET train dir not found: {det_train}")
    
    # 2. Convert MOT Val to YOLO (for proper Validation metrics)
    if os.path.exists(mot_val):
        convert_mot_to_yolo(mot_val, yolo_out)
    else:
        print(f"MOT val dir not found: {mot_val}")
        
    # 3. Convert MOT Val to Standard MOT (for Tracking eval)
    if os.path.exists(mot_val):
        convert_mot_to_standard(mot_val, mot_out)
    else:
        print(f"MOT val dir not found: {mot_val}")
