#!/usr/bin/env python3
"""
Preprocess Semantics for RobotCycle Dataset
============================================
Runs Grounding DINO + SAM2 on every frame in the dataset to produce:
  1. A per-pixel semantic label map  (uint8, H x W)  -> semantics/<frame>.npy
  2. A per-pixel drivability score   (float16, H x W) -> drivability/<frame>.npy

Usage (inside Docker container):
    python preprocess_semantics.py --data_root /data_224
    python preprocess_semantics.py --data_root /data_224 --stride 3 --viz
    python preprocess_semantics.py --data_root /data_224 --seq seq_001 seq_002

Multi-GPU (2x GPUs):
    python preprocess_semantics.py --data_root /data_224 --num_gpus 2

The script is resumable: it skips frames that already have a .npy file in semantics/.
"""

import os
import warnings
warnings.filterwarnings("ignore", message=".*antialias.*")
# warnings.filterwarnings("ignore", category=UserWarning)
import sys
import glob
import argparse
import time
import numpy as np
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
from threading import Thread
from queue import Queue
import torch.multiprocessing as mp

import sdpa_compat  # noqa: F401  — patches torch for PyTorch < 2.0 compatibility

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert

# ─────────────────────────────────────────────────────────────
# Semantic categories (order = class index in label map)
# Priority: earlier in list wins when masks overlap
# ─────────────────────────────────────────────────────────────
CATEGORIES = [
    "road",         # 0  - drivable
    "sidewalk",     # 1  - marginal
    "car",          # 2  - obstacle
    "bus",          # 3  - obstacle
    "vehicle",      # 4  - obstacle (catch-all)
    "cyclist",      # 5  - obstacle
    "pedestrian",   # 6  - obstacle
    "building",     # 7  - obstacle
    "tree",         # 8  - obstacle
    "bush",         # 9  - obstacle
    "sky",          # 10 - background
]

# Drivability score for each class (1.0 = safe, 0.0 = collision)
DRIVABILITY = {
    "road":         1.0,
    "sidewalk":     0.0,
    "car":          0.0,
    "bus":          0.0,
    "vehicle":      0.0,
    "cyclist":      0.0,
    "pedestrian":   0.0,
    "building":     0.0,
    "tree":         0.0,
    "bush":         0.0,
    "sky":          0.5,
}

# Class index for "unlabelled" pixels (not matched by any detection)
UNLABELLED_IDX = 255
UNLABELLED_DRIVABILITY = 0.5  # ambiguous

# Build the Grounding DINO text prompt (each category ends with a dot)
TEXT_PROMPT = ". ".join(CATEGORIES) + "."


def build_models(device, sam2_checkpoint, sam2_config, gdino_config, gdino_checkpoint):
    """Load SAM2 and Grounding DINO models once."""
    print(f"[{device}] Loading SAM2 from {sam2_checkpoint}...")
    sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    print(f"[{device}] Loading Grounding DINO from {gdino_checkpoint}...")
    grounding_model = load_model(
        model_config_path=gdino_config,
        model_checkpoint_path=gdino_checkpoint,
        device=device
    )
    return sam2_predictor, grounding_model


def segment_frame(img_path, sam2_predictor, grounding_model, device,
                  box_threshold=0.35, text_threshold=0.25, preloaded_image=None):
    """
    Run Grounding DINO + SAM2 on a single image.
    
    Args:
        preloaded_image: tuple (image_source, image) from prefetch thread, or None.
    
    Returns:
        masks: (N, H, W) bool array
        class_names: list of str (one per mask)
        confidences: list of float
        dims: (h, w)
    """
    if preloaded_image is not None:
        image_source, image = preloaded_image
    else:
        image_source, image = load_image(img_path)
    h, w, _ = image_source.shape

    # Grounding DINO detection
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device
    )

    if len(boxes) == 0:
        return np.zeros((0, h, w), dtype=bool), [], [], (h, w)

    # Convert boxes to pixel coordinates
    boxes = boxes * torch.Tensor([w, h, w, h]).to(boxes.device)
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

    # SAM2 segmentation
    sam2_predictor.set_image(image_source)
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # Squeeze if needed: (N, 1, H, W) -> (N, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    return masks.astype(bool), labels, confidences.cpu().numpy().tolist(), (h, w)


def masks_to_labelmap(masks, class_names, confidences, h, w):
    """
    Merge per-instance masks into a single semantic label map.
    
    Priority resolution:
      - For each pixel, if multiple masks overlap, prefer:
        1. Lower category index (CATEGORIES order = priority)
        2. Higher confidence within same category
    
    Returns:
        label_map: (H, W) uint8, values are indices into CATEGORIES or UNLABELLED_IDX
        drivability_map: (H, W) float16
    """
    label_map = np.full((h, w), UNLABELLED_IDX, dtype=np.uint8)
    priority_map = np.full((h, w), len(CATEGORIES) + 1, dtype=np.int32)  # lower = higher priority

    # Build a lookup: class_name -> index in CATEGORIES
    cat_to_idx = {}
    for i, cat in enumerate(CATEGORIES):
        cat_to_idx[cat] = i

    for mask, class_name, conf in zip(masks, class_names, confidences):
        # Map detected label to our category index
        cls_idx = cat_to_idx.get(class_name, None)
        if cls_idx is None:
            # Fuzzy match: check if any category is a substring of the detection
            for cat, idx in cat_to_idx.items():
                if cat in class_name or class_name in cat:
                    cls_idx = idx
                    break
        if cls_idx is None:
            continue  # skip unknown detections

        # Only overwrite pixels if this detection has higher priority (lower index)
        overwrite = mask & (cls_idx < priority_map)
        label_map[overwrite] = cls_idx
        priority_map[overwrite] = cls_idx

    # Build drivability map
    drivability_map = np.full((h, w), UNLABELLED_DRIVABILITY, dtype=np.float32)
    for i, cat in enumerate(CATEGORIES):
        drivability_map[label_map == i] = DRIVABILITY[cat]

    # Smooth the drivability map (creates a gradient around obstacles)
    # Use a kernel size relative to image resolution (e.g. 1/16 of width)
    ksize = int(w / 16)
    if ksize % 2 == 0:
        ksize += 1
    drivability_map = cv2.GaussianBlur(drivability_map, (ksize, ksize), 0)
    
    return label_map, drivability_map.astype(np.float16)


def save_visualisation(img_path, label_map, out_path, image=None):
    """Save a colour-coded overlay of the semantic labels on the original image."""
    PALETTE = {
        0:  (128, 128, 128),  # road - grey
        1:  (180, 130, 70),   # sidewalk - slate blue
        2:  (0, 0, 255),      # car - red
        3:  (0, 0, 200),      # bus - dark red
        4:  (0, 0, 180),      # vehicle - darker red
        5:  (0, 255, 255),    # cyclist - yellow
        6:  (0, 165, 255),    # pedestrian - orange
        7:  (100, 100, 100),  # building - dark grey
        8:  (0, 180, 0),      # tree - green
        9:  (34, 139, 34),    # bush - forest green
        10: (255, 200, 150),  # sky - light blue
    }

    if image is None:
        img = cv2.imread(str(img_path))
    else:
        img = image.copy()
        
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
    overlay = np.zeros_like(img)
    for cls_idx, colour in PALETTE.items():
        overlay[label_map == cls_idx] = colour
    
    blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    cv2.imwrite(str(out_path), blended)


def save_drivability_visualisation(img_path, drivability_map, out_path, image=None):
    """Save a colour-coded overlay of the drivability map on the original image."""
    if image is None:
        img = cv2.imread(str(img_path))
    else:
        img = image.copy()
        
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Map drivability [0.0, 1.0] to a colormap (e.g. Jet)
    # cv2.COLORMAP_JET: 0 is Blue, 255 is Red.
    # We want 1.0 (drivable) to be Blue, and 0.0 (obstacle) to be Red, so we invert.
    score_scaled = ((1.0 - drivability_map) * 255.0).clip(0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(score_scaled, cv2.COLORMAP_JET)
    
    # Blend with original image
    blended = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(str(out_path), blended)


# ─────────────────────────────────────────────────────────────
# Image prefetching (overlaps disk I/O with GPU computation)
# ─────────────────────────────────────────────────────────────
def prefetch_worker(file_queue, result_queue, max_prefetch=8):
    """Background thread that pre-loads images from disk."""
    while True:
        item = file_queue.get()
        if item is None:  # poison pill
            result_queue.put(None)
            break
        img_path, fname = item
        try:
            image_data = load_image(img_path)
            result_queue.put((img_path, fname, image_data))
        except Exception as e:
            result_queue.put((img_path, fname, None))


def process_sequence(seq_path, sam2_predictor, grounding_model, device, args,
                     output_224_seq_path=None):
    """Process all frames in one sequence folder with image prefetching.
    
    If output_224_seq_path is provided, also saves downscaled 224x224 maps there.
    """
    seq_name = os.path.basename(seq_path)
    image_dir = os.path.join(seq_path, "images")
    sem_dir = os.path.join(seq_path, "semantics")
    drv_dir = os.path.join(seq_path, "drivability")
    
    if not os.path.isdir(image_dir):
        print(f"  [{device}] Skipping {seq_name}: no images/ directory")
        return 0

    os.makedirs(sem_dir, exist_ok=True)
    os.makedirs(drv_dir, exist_ok=True)
    if args.viz:
        sem_vis_dir = os.path.join(seq_path, "semantics_vis")
        drv_vis_dir = os.path.join(seq_path, "drivability_vis")
        os.makedirs(sem_vis_dir, exist_ok=True)
        os.makedirs(drv_vis_dir, exist_ok=True)

    # Optional 224x224 output directories
    sem_dir_224 = None
    drv_dir_224 = None
    if output_224_seq_path:
        sem_dir_224 = os.path.join(output_224_seq_path, "semantics")
        drv_dir_224 = os.path.join(output_224_seq_path, "drivability")
        os.makedirs(sem_dir_224, exist_ok=True)
        os.makedirs(drv_dir_224, exist_ok=True)
        
        if args.viz:
            sem_vis_dir_224 = os.path.join(output_224_seq_path, "semantics_vis")
            drv_vis_dir_224 = os.path.join(output_224_seq_path, "drivability_vis")
            os.makedirs(sem_vis_dir_224, exist_ok=True)
            os.makedirs(drv_vis_dir_224, exist_ok=True)

    img_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    
    # Apply stride
    if args.stride > 1:
        img_files = img_files[::args.stride]

    # Filter to only frames that need processing (for accurate progress bar)
    work_items = []
    skipped = 0
    for img_path in img_files:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(sem_dir, f"{fname}.npy")
        drv_path = os.path.join(drv_dir, f"{fname}.npy")
        if os.path.exists(label_path) and os.path.exists(drv_path) and not args.force:
            skipped += 1
        else:
            work_items.append((img_path, fname))

    if not work_items:
        if skipped > 0:
            print(f"  [{device}] {seq_name}: all {skipped} frames already done, skipping")
        return 0

    # Start prefetch thread — feed it the work items directly
    result_queue = Queue(maxsize=8)

    def prefetch_thread_fn():
        for img_path, fname in work_items:
            try:
                image_data = load_image(img_path)
                result_queue.put((img_path, fname, image_data))
            except Exception:
                result_queue.put((img_path, fname, None))
        result_queue.put(None)  # poison pill

    prefetch_thread = Thread(target=prefetch_thread_fn, daemon=True)
    prefetch_thread.start()

    processed = 0
    pbar = tqdm(total=len(work_items), desc=f"  [{device}] {seq_name}", leave=False)
    
    while True:
        result = result_queue.get()
        if result is None:
            break

        img_path, fname, preloaded = result
        label_path = os.path.join(sem_dir, f"{fname}.npy")
        drv_path = os.path.join(drv_dir, f"{fname}.npy")

        try:
            masks, class_names, confidences, (h, w) = segment_frame(
                img_path, sam2_predictor, grounding_model, device,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                preloaded_image=preloaded
            )

            label_map, drivability_map = masks_to_labelmap(
                masks, class_names, confidences, h, w
            )

            np.save(label_path, label_map)
            np.save(drv_path, drivability_map)

            # Save downscaled 224x224 versions if requested
            if sem_dir_224 and drv_dir_224:
                label_224 = cv2.resize(label_map, (224, 224), interpolation=cv2.INTER_NEAREST)
                drv_224 = cv2.resize(drivability_map, (224, 224), interpolation=cv2.INTER_NEAREST)
                np.save(os.path.join(sem_dir_224, f"{fname}.npy"), label_224)
                np.save(os.path.join(drv_dir_224, f"{fname}.npy"), drv_224)

            if args.viz:
                vis_sem_path = os.path.join(sem_vis_dir, f"{fname}_sem.png")
                vis_drv_path = os.path.join(drv_vis_dir, f"{fname}_drv.png")
                
                # To avoid re-reading the image multiple times, load it once
                orig_img = cv2.imread(str(img_path))
                save_visualisation(None, label_map, vis_sem_path, image=orig_img)
                save_drivability_visualisation(None, drivability_map, vis_drv_path, image=orig_img)
                
                if sem_dir_224 and drv_dir_224:
                    vis_sem_path_224 = os.path.join(sem_vis_dir_224, f"{fname}_sem.png")
                    vis_drv_path_224 = os.path.join(drv_vis_dir_224, f"{fname}_drv.png")
                    img_224 = cv2.resize(orig_img, (224, 224), interpolation=cv2.INTER_AREA)
                    save_visualisation(None, label_224, vis_sem_path_224, image=img_224)
                    save_drivability_visualisation(None, drv_224, vis_drv_path_224, image=img_224)

            processed += 1
        except Exception as e:
            print(f"  [{device}] Error processing {fname}: {e}")

        pbar.update(1)
    
    pbar.close()
    prefetch_thread.join()

    total = processed + skipped
    if skipped > 0:
        print(f"  [{device}] {seq_name}: processed {processed}, skipped {skipped} (already done)")
    else:
        print(f"  [{device}] {seq_name}: processed {processed} frames")
    return processed


# ─────────────────────────────────────────────────────────────
# Multi-GPU worker (one process per GPU)
# ─────────────────────────────────────────────────────────────
def gpu_worker(gpu_id, seq_paths, args, return_dict):
    """Worker function for multi-GPU processing. Runs in a separate process."""
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    
    # AMP setup for this GPU
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    # Load models on this GPU
    sam2_predictor, grounding_model = build_models(
        device, args.sam2_checkpoint, args.sam2_config,
        args.gdino_config, args.gdino_checkpoint
    )

    total = 0
    for seq_path in seq_paths:
        if os.path.isdir(seq_path):
            seq_name = os.path.basename(seq_path)
            out_224 = os.path.join(args.data_root_224, seq_name) if args.data_root_224 else None
            total += process_sequence(seq_path, sam2_predictor, grounding_model, device, args,
                                     output_224_seq_path=out_224)
    
    return_dict[gpu_id] = total


def main():
    parser = argparse.ArgumentParser(description="Preprocess semantic segmentation for RobotCycle dataset")
    parser.add_argument("--data_root", type=str, default="/data",
                        help="Path to dataset root (e.g. /data_224)")
    parser.add_argument("--seq", nargs="*", default=None,
                        help="Specific sequence folders to process (default: all)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Process every Nth frame (default: 1 = all frames)")
    parser.add_argument("--num_gpus", type=int, default=2,
                        help="Number of GPUs to use (default: 2). Sequences are split across GPUs.")
    parser.add_argument("--box_threshold", type=float, default=0.35,
                        help="Grounding DINO box confidence threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25,
                        help="Grounding DINO text confidence threshold")
    parser.add_argument("--viz", action="store_true",
                        help="Save colour-coded overlay images to semantics_vis/")
    parser.add_argument("--force", action="store_true",
                        help="Re-process frames even if output already exists")
    parser.add_argument("--data_root_224", type=str, default="/data_224",
                        help="Also save downscaled 224x224 maps here (e.g. /data_224). "
                             "Same folder structure as data_root.")
    
    # Model paths (defaults match repo structure)
    parser.add_argument("--sam2_checkpoint", type=str,
                        default="./checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_config", type=str,
                        default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--gdino_config", type=str,
                        default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--gdino_checkpoint", type=str,
                        default="gdino_checkpoints/groundingdino_swint_ogc.pth")

    args = parser.parse_args()

    # Validate
    if not os.path.isdir(args.data_root):
        print(f"Error: {args.data_root} is not a directory")
        sys.exit(1)

    # Find sequences
    if args.seq:
        seq_folders = [os.path.join(args.data_root, s) for s in args.seq]
    else:
        seq_folders = sorted([
            os.path.join(args.data_root, d)
            for d in os.listdir(args.data_root)
            if os.path.isdir(os.path.join(args.data_root, d))
        ])

    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    
    print(f"\nPreprocess Semantics")
    print(f"  Data root: {args.data_root}")
    if args.data_root_224:
        print(f"  Also saving 224x224 maps to: {args.data_root_224}")
    print(f"  Sequences: {len(seq_folders)}")
    print(f"  Stride: {args.stride} | Box threshold: {args.box_threshold} | Text threshold: {args.text_threshold}")
    print(f"  GPUs: {num_gpus} | Visualise: {args.viz} | Force: {args.force}")
    print()

    # Print category -> drivability mapping
    print("Category -> Drivability mapping:")
    for i, cat in enumerate(CATEGORIES):
        print(f"  [{i:2d}] {cat:<15s} -> {DRIVABILITY[cat]:.1f}")
    print(f"  [{UNLABELLED_IDX:3d}] {'unlabelled':<15s} -> {UNLABELLED_DRIVABILITY:.1f}")
    print()

    t0 = time.time()

    if num_gpus > 1:
        # ── Multi-GPU: split sequences round-robin across GPUs ──
        gpu_seqs = [[] for _ in range(num_gpus)]
        for i, seq in enumerate(seq_folders):
            gpu_seqs[i % num_gpus].append(seq)

        for gpu_id in range(num_gpus):
            n = len(gpu_seqs[gpu_id])
            print(f"  GPU {gpu_id} ({torch.cuda.get_device_name(gpu_id)}): {n} sequences")

        print()
        mp.set_start_method("spawn", force=True)
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []

        for gpu_id in range(num_gpus):
            p = mp.Process(target=gpu_worker, args=(gpu_id, gpu_seqs[gpu_id], args, return_dict))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        total_processed = sum(return_dict.values())

    else:
        # ── Single GPU ──
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda"):
            torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU (this will be very slow)")

        sam2_predictor, grounding_model = build_models(
            device, args.sam2_checkpoint, args.sam2_config,
            args.gdino_config, args.gdino_checkpoint
        )

        total_processed = 0
        for seq_path in seq_folders:
            if os.path.isdir(seq_path):
                seq_name = os.path.basename(seq_path)
                out_224 = os.path.join(args.data_root_224, seq_name) if args.data_root_224 else None
                total_processed += process_sequence(
                    seq_path, sam2_predictor, grounding_model, device, args,
                    output_224_seq_path=out_224
                )

    elapsed = time.time() - t0
    fps = total_processed / elapsed if elapsed > 0 else 0
    print(f"\nDone! Processed {total_processed} frames in {elapsed:.1f}s ({fps:.1f} fps)")


if __name__ == "__main__":
    main()
