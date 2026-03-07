import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import cv2
from PIL import Image
import csv
import sys

# directories
IMG_INPUT_DIR = "robotcycle/img_in/"
JSON_DIR = "robotcycle/gs2_json_out/"
IMG_MASK_OUT_DIR = "robotcycle/gs2_img_out/"
IMG_OUT_DIR = "robotcycle/img_out10_test_cuda/"
CSV_IN_PATH = "robotcycle/gaze_csv/adjusted_points_v3sample2.csv"
CSV_OUT_PATH = "robotcycle/gaze_csv/adjusted_points_v3sample2ann.csv"
POINT_RADIUS = 10  # radius around gaze point to consider for annotation
MAX_IMAGES = 2  # Set the maximum number of images to process
TIME_START = 1730815570.0  # starting timestamp (inclusive)
TIME_END = 9999999999.0  # ending timestamp (inclusive)
HEATMAP_PATH = "robotcycle/heatmaps1/"
HEATMAP_ALPHA = 0.3
SAVE_HEATMAP_PNG = False

# categories in decreasing priority order
CATEGORIES = ["cyclist", "pedestrian", "traffic sign", "car", "bus", "vehicle", "sidewalk", "road", "building", "tree", "sky"]
TEXT_PROMPT = ". ".join([f"{cat}." for cat in CATEGORIES])

# Redirect terminal outputs to a log file (robust)
os.makedirs(IMG_OUT_DIR, exist_ok=True)
log_file_path = os.path.join(IMG_OUT_DIR, "main.log")
# open in append mode, line-buffered, UTF-8
log_file = open(log_file_path, "a", buffering=1, encoding="utf-8")
# redirect Python-level streams
sys.stdout = log_file
sys.stderr = log_file
# also redirect underlying file descriptors so native/C libs are captured
os.dup2(log_file.fileno(), 1)
os.dup2(log_file.fileno(), 2)
# ensure periodic flushing where important: use sys.stdout.flush() after critical prints

def get_segments(timestamp):
    """
    Given a timestamp, runs Grounded-SAM 2 to get segmentations and saves the results.
    timestamp: str or float, timestamp of the image to process
    """

    # Hyper parameters
    img_path = Path(f"{IMG_INPUT_DIR}{timestamp}.png")
    SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
    SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    json_output_dir = Path(JSON_DIR)
    img_output_dir = Path(IMG_OUT_DIR)
    VISUALISE = True
    DUMP_JSON_RESULTS = True
    MULTIMASK_OUTPUT = False

    # environment settings
    # use bfloat16

    # build SAM2 image predictor
    sam2_checkpoint = SAM2_CHECKPOINT
    model_cfg = SAM2_MODEL_CONFIG
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino model
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG, 
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE
    )


    # setup the input image and text prompt for SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot
    text = TEXT_PROMPT

    image_source, image = load_image(img_path)

    sam2_predictor.set_image(image_source)

    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE
    )

    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


    # FIXME: figure how does this influence the G-DINO model
    torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=MULTIMASK_OUTPUT,
    )

    # Sample the best mask according to the score
    if MULTIMASK_OUTPUT:
        best = np.argmax(scores, axis=1)                     
        masks = masks[np.arange(masks.shape[0]), best]       

    # Post-process the output of the model to get the masks, scores, and logits for visualization
    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)


    confidences = confidences.numpy().tolist()
    class_names = labels

    class_ids = np.array(list(range(len(class_names))))

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    # Visualize image with supervision useful API
    if VISUALISE:
        img = cv2.imread(img_path)
        detections = sv.Detections(
            xyxy=input_boxes,  # (n, 4)
            mask=masks.astype(bool),  # (n, h, w)
            class_id=class_ids
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        # cv2.imwrite(os.path.join(IMG_MASK_OUT_DIR, f"{timestamp}_annotated.jpg"), annotated_frame)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(os.path.join(IMG_MASK_OUT_DIR, f"{timestamp}_mask.jpg"), annotated_frame)

    # Dump the results in standard format and save as json files
    if DUMP_JSON_RESULTS:
        # convert mask into rle format
        mask_rles = [single_mask_to_rle(mask) for mask in masks]

        input_boxes = input_boxes.tolist()
        scores = scores.tolist()
        # save the results in standard format
        results = {
            "image_path": img_path,
            "annotations" : [
                {
                    "class_name": class_name,
                    "bbox": box,
                    "segmentation": mask_rle,
                    "score": score,
                }
                for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
            ],
            "box_format": "xyxy",
            "img_width": w,
            "img_height": h,
        }

        with open(os.path.join(json_output_dir, f"{timestamp}.json"), "w") as f:
            results["image_path"] = str(img_path)  # Convert PosixPath to string
            json.dump(results, f, indent=4)


def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def query_point_annotations(json_path, x, y):
    """
    x,y: ints (pixel coords, origin top-left)
    returns: list of matches sorted by descending score:
      [{'class_name':..., 'score':..., 'bbox':..., 'ann_index':..., 'mask_contains': True}, ...]
    """
    data = json.load(open(json_path, 'r'))
    H = data['img_height']
    W = data['img_width']
    if not (0 <= x < W and 0 <= y < H):
        return []

    matches = []
    for i, ann in enumerate(data.get('annotations', [])):
        seg = ann['segmentation']
        # COCO RLE expects counts as bytes
        rle = {"size": seg["size"], "counts": seg["counts"].encode("utf-8")}
        mask = mask_util.decode(rle)  # numpy array (H, W), dtype=uint8
        contains = bool(mask[y, x])
        bbox_contains = False
        if 'bbox' in ann:
            x1, y1, x2, y2 = ann['bbox']
            bbox_contains = (x1 <= x <= x2) and (y1 <= y <= y2)
        matches.append({
            "ann_index": i,
            "class_name": ann.get("class_name"),
            "score": float(ann.get("score", [0])[0]) if isinstance(ann.get("score"), list) else float(ann.get("score", 0)),
            "bbox": ann.get("bbox"),
            "mask_contains": contains,
            "bbox_contains": bbox_contains
        })

    # prefer mask hits then bbox hits; sort by score descending
    matches = [m for m in matches if m["mask_contains"] or m["bbox_contains"]]
    matches.sort(key=lambda m: (m["mask_contains"], m["score"]), reverse=True)
    return matches


def get_best_annotation(timestamp, x, y):
    """
    Given a timestamp and point (x, y), returns the best matching annotation.
    Will query all points within a radius around (x, y).
    Returns the class_name of the highest priority match, or "None" if no match.
    """
    json_path = f"{JSON_DIR}{timestamp}.json"
    points_to_check = []
    for dx in range(-POINT_RADIUS, POINT_RADIUS + 1):
        for dy in range(-POINT_RADIUS, POINT_RADIUS + 1):
            if dx*dx + dy*dy <= POINT_RADIUS*POINT_RADIUS:
                points_to_check.append((x + dx, y + dy))
    matches = []
    for px, py in points_to_check:
        matches.extend(query_point_annotations(json_path, px, py))
    if matches:
        return matches[0]['class_name']
    else:
        return "None"


def annotate_image(timestamp, x, y, best_annotation):
    """
    Plots a point on the image at (x, y) and saves the result.
    """
    img_path = os.path.join(IMG_INPUT_DIR, f"{timestamp}.png")
    output_path = os.path.join(IMG_OUT_DIR, f"{timestamp}.png")
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")

    # Load the heatmap
    # Load and overlay heatmap if available
    heatmap_file = os.path.join(HEATMAP_PATH, f"{timestamp}.npy")
    if os.path.exists(heatmap_file):
        try:
            heat = np.load(heatmap_file)  # expected 2D array of values
            if heat is None:
                raise ValueError("Loaded heatmap is None")

            # Ensure 2D
            if heat.ndim > 2:
                heat = np.squeeze(heat)
            if heat.ndim != 2:
                raise ValueError(f"Heatmap must be 2D, got ndim={heat.ndim}")

            # Resize heatmap to image size if necessary
            img_h, img_w = img.shape[:2]
            if (heat.shape[0], heat.shape[1]) != (img_h, img_w):
                heat_resized = cv2.resize(heat.astype(np.float32), (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            else:
                heat_resized = heat.astype(np.float32)

            # Normalize to 0-255
            mn, mx = float(np.nanmin(heat_resized)), float(np.nanmax(heat_resized))
            if np.isclose(mx, mn):
                heat_norm = np.zeros_like(heat_resized, dtype=np.uint8)
            else:
                heat_norm = np.clip((heat_resized - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)

            # Apply a colormap
            heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_HOT)  # BGR

            # Blend heatmap with image
            alpha = float(HEATMAP_ALPHA)
            img = cv2.addWeighted(img, 1.0 - alpha, heat_color, alpha, 0)

            # Optionally save a visualization of the heatmap alone
            if SAVE_HEATMAP_PNG:
                try:
                    heat_vis_path = os.path.join(HEATMAP_PATH, f"{timestamp}_heatmap.png")
                    cv2.imwrite(heat_vis_path, heat_color)
                except Exception:
                    pass

        except Exception:
            # If anything goes wrong loading/processing heatmap, continue without overlay
            pass
    else:
        # No heatmap file found; continue without overlay
        pass

    # Draw the point on the image
    color = (0, 255, 0)  # Green color in BGR
    radius = POINT_RADIUS
    thickness = 2
    cv2.circle(img, (x, y), radius, color, thickness)

    # Add timestamp and best annotation text
    timestamp_text = f"{timestamp}"
    annotation_text = f"{best_annotation}"

    # Add background to the text
    timestamp_font_scale = 0.8
    annotation_font_scale = 3
    font_thickness = 2

    # Calculate text sizes
    timestamp_size, _ = cv2.getTextSize(timestamp_text, cv2.FONT_HERSHEY_SIMPLEX, timestamp_font_scale, font_thickness)
    annotation_size, _ = cv2.getTextSize(annotation_text, cv2.FONT_HERSHEY_SIMPLEX, annotation_font_scale, font_thickness)

    timestamp_width, timestamp_height = timestamp_size
    annotation_width, annotation_height = annotation_size

    # Position of the text
    text_x, text_y = 10, 30  # Starting position for timestamp
    annotation_y = text_y + timestamp_height + 80  # Position for annotation below timestamp

    background_color = (0, 0, 0)  # Black background
    text_color = (255, 255, 255)  # White text

    # Draw the background rectangle for timestamp
    cv2.rectangle(img, (text_x, text_y - timestamp_height - 5), 
                  (text_x + max(timestamp_width, annotation_width), annotation_y + annotation_height + 5), 
                  background_color, -1)

    # Draw the timestamp
    cv2.putText(img, timestamp_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, timestamp_font_scale, text_color, font_thickness)

    # Draw the annotation
    cv2.putText(img, annotation_text, (text_x, annotation_y), cv2.FONT_HERSHEY_SIMPLEX, annotation_font_scale, text_color, font_thickness)

    # Save the modified image
    cv2.imwrite(output_path, img)

def img_to_video(fps=10, out_name="output_video.mp4"):
    # create video from PNGs in IMG_OUT_DIR and save to IMG_OUT_DIR/out_name
    files = sorted([f for f in os.listdir(IMG_OUT_DIR) if f.endswith(".png")])
    if not files:
        print("No PNGs found in", IMG_OUT_DIR)
        return
    first_path = os.path.join(IMG_OUT_DIR, files[0])
    first = cv2.imread(first_path)
    if first is None:
        print("Failed to read first image:", first_path)
        return
    height, width = first.shape[:2]
    out_path = os.path.join(IMG_OUT_DIR, out_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # widely-compatible mp4 codec
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    for filename in files:
        img_path = os.path.join(IMG_OUT_DIR, filename)
        img = cv2.imread(img_path)
        if img is None:
            print("skip unreadable image:", img_path)
            continue
        out.write(img)
    out.release()
    print("Saved video to", out_path)

# Load csv with timestamps and projected gaze points
with open(CSV_IN_PATH, "r") as f:
    reader = csv.DictReader(f)
    gaze_data = {row["timestamp"]: (float(row["projection_point_x"]), float(row["projection_point_y"]), "None") for row in reader}

# Process each image in IMG_INPUT_DIR
processed_count = 0

# iterate images in numerically ascending order (falls back to float/int parse, then string)
files = [f for f in os.listdir(IMG_INPUT_DIR) if f.lower().endswith(".png")]
def _numeric_key(fname):
    base = os.path.splitext(fname)[0]
    try:
        return int(base)
    except Exception:
        try:
            return float(base)
        except Exception:
            return base  # fallback to string sort

files = sorted(files, key=_numeric_key)
try:
    for img_file in files:
        if processed_count >= MAX_IMAGES:
            break

        timestamp = os.path.splitext(img_file)[0]
        t_float = float(timestamp)
        if t_float < TIME_START or t_float > TIME_END:
            continue

        print(f"Processing image {img_file} ({processed_count + 1} out of {MAX_IMAGES})")
        # print(f"Processing image {img_file} ({processed_count + 1} out of {MAX_IMAGES})", end="")

        # ensure we have gaze data for this timestamp
        if timestamp not in gaze_data:
            print(" - no gaze data, skipping")
            continue

        get_segments(timestamp)

        # Get gaze point from csv
        x, y, _ = gaze_data[timestamp]
        x_int = int(round(x))
        y_int = int(round(y))

        # Get best annotation for the gaze point
        best_annotation = get_best_annotation(timestamp, x_int, y_int)

        # Add the best annotation to the CSV data - store as flat values
        gaze_data[timestamp] = (x, y, best_annotation)

        # Annotate image with gaze point, timestamp, best annotation
        annotate_image(timestamp, x_int, y_int, best_annotation)

        processed_count += 1
        
except Exception as e:
    print("Error during processing:", str(e))

# Save the updated gaze data with annotations to a new CSV file
try:
    with open(CSV_OUT_PATH, "w", newline="") as f:
        fieldnames = ["timestamp", "x", "y", "best_annotation"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for timestamp, data in gaze_data.items():
            x, y, best_annotation = data
            writer.writerow({"timestamp": timestamp, "x": x, "y": y, "best_annotation": best_annotation})
except Exception as e:
    print("Error during CSV writing:", str(e))

# Create video from annotated images
print("\nCreating video from annotated images...")
try:
    img_to_video(fps=10)
    print(" Done.")
except Exception as e:
    print("Error during video creation:", str(e))