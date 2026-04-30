#!/usr/bin/env python3
"""
Preprocess Semantics for RobotCycle Dataset (Video Tracking Version)
====================================================================
Runs Grounding DINO on keyframes and uses SAM 2 Video Predictor to temporally 
track obstacles. This provides much cleaner, temporally consistent masks and 
is significantly faster than running Grounding DINO per-frame.

To handle very long sequences (e.g. 20k frames), we split them into manageable 
chunks (e.g. 150 frames) transparently.

Usage (inside Docker container):
    python preprocess_semantics_video.py --data_root /data_224
    python preprocess_semantics_video.py --data_root /data_224 --keyframe_interval 5
"""

import os
import warnings
warnings.filterwarnings("ignore", message=".*antialias.*")
import sys
import glob
import argparse
import time
import numpy as np
import torch
import cv2
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm
import torch.multiprocessing as mp

import sdpa_compat  # noqa: F401

from sam2.build_sam import build_sam2_video_predictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert

# ─────────────────────────────────────────────────────────────
# Semantic categories (order = class index in label map)
# Expanded ontology for better bounding box anchoring
# ─────────────────────────────────────────────────────────────
CATEGORIES = [
    "road",         # 0  - drivable
    "sidewalk",     # 1  - marginal
    "car",          # 2  - obstacle
    "bus",          # 3  - obstacle
    "van",          # 4  - obstacle
    "truck",        # 5  - obstacle
    "vehicle",      # 6  - obstacle (catch-all)
    "motorcycle",   # 7  - obstacle
    "cyclist",      # 8  - obstacle
    "pedestrian",   # 9  - obstacle
    "building",     # 10 - obstacle
    "tree",         # 11 - obstacle
    "bush",         # 12 - obstacle
    "sky",          # 13 - background
    "grass",        # 14 - rough terrain / background
    "pole",         # 15 - obstacle
    "traffic sign", # 16 - obstacle
]

# Drivability score for each class (1.0 = safe, 0.0 = collision)
DRIVABILITY = {
    "road":         1.0,
    "sidewalk":     0.0,
    "car":          0.0,
    "bus":          0.0,
    "van":          0.0,
    "truck":        0.0,
    "vehicle":      0.0,
    "motorcycle":   0.0,
    "cyclist":      0.0,
    "pedestrian":   0.0,
    "building":     0.0,
    "tree":         0.0,
    "bush":         0.0,
    "sky":          0.5,  # ambiguous
    "grass":        0.0,  
    "pole":         0.0,
    "traffic sign": 0.0,
}

UNLABELLED_IDX = 255
UNLABELLED_DRIVABILITY = 0.5  # segmentation should be decent

TEXT_PROMPT = ". ".join(CATEGORIES) + "."


def build_models(device, sam2_checkpoint, sam2_config, gdino_config, gdino_checkpoint):
    """Load SAM2 Video Predictor and Grounding DINO models once."""
    print(f"[{device}] Loading SAM2 Video Predictor from {sam2_checkpoint}...")
    video_predictor = build_sam2_video_predictor(sam2_config, sam2_checkpoint, device=device)

    print(f"[{device}] Loading Grounding DINO from {gdino_checkpoint}...")
    grounding_model = load_model(
        model_config_path=gdino_config,
        model_checkpoint_path=gdino_checkpoint,
        device=device
    )
    return video_predictor, grounding_model


def masks_to_labelmap(masks, class_names, confidences, h, w):
    """Merge per-instance masks into a single semantic label map and drivability map."""
    label_map = np.full((h, w), UNLABELLED_IDX, dtype=np.uint8)
    priority_map = np.full((h, w), len(CATEGORIES) + 1, dtype=np.int32)

    cat_to_idx = {cat: i for i, cat in enumerate(CATEGORIES)}

    for mask, class_name, _ in zip(masks, class_names, confidences):
        cls_idx = cat_to_idx.get(class_name, None)
        if cls_idx is None:
            for cat, idx in cat_to_idx.items():
                if cat in class_name or class_name in cat:
                    cls_idx = idx
                    break
        if cls_idx is None:
            continue

        overwrite = mask & (cls_idx < priority_map)
        label_map[overwrite] = cls_idx
        priority_map[overwrite] = cls_idx

    drivability_map = np.full((h, w), UNLABELLED_DRIVABILITY, dtype=np.float32)
    for i, cat in enumerate(CATEGORIES):
        drivability_map[label_map == i] = DRIVABILITY[cat]

    ksize = int(w / 16)
    if ksize % 2 == 0:
        ksize += 1
    drivability_map = cv2.GaussianBlur(drivability_map, (ksize, ksize), 0)
    
    return label_map, drivability_map.astype(np.float16)


def save_visualisation(img_path, label_map, out_path, image=None):
    # Palette maps indices to colors (BGR) for all 17 classes
    PALETTE = [
        (128, 128, 128),  # road
        (180, 130, 70),   # sidewalk
        (0, 0, 255),      # car
        (0, 0, 200),      # bus
        (0, 0, 230),      # van
        (0, 0, 180),      # truck
        (30, 0, 180),     # vehicle
        (50, 255, 255),   # motorcycle
        (0, 255, 255),    # cyclist
        (0, 165, 255),    # pedestrian
        (100, 100, 100),  # building
        (0, 180, 0),      # tree
        (34, 139, 34),    # bush
        (255, 200, 150),  # sky
        (0, 200, 50),     # grass
        (50, 50, 50),     # pole
        (0, 0, 150),      # traffic sign
    ]
    
    if image is None:
        img = cv2.imread(str(img_path))
    else:
        img = image.copy()
        
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
    overlay = np.zeros_like(img)
    for cls_idx, colour in enumerate(PALETTE):
        overlay[label_map == cls_idx] = colour
    
    blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
    cv2.imwrite(str(out_path), blended)


def save_drivability_visualisation(img_path, drivability_map, out_path, image=None):
    if image is None:
        img = cv2.imread(str(img_path))
    else:
        img = image.copy()
        
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    score_scaled = ((1.0 - drivability_map) * 255.0).clip(0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(score_scaled, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(str(out_path), blended)


def process_sequence(seq_path, video_predictor, grounding_model, device, args, output_224_seq_path=None):
    """Processes sequence with SAM 2 Video Predictor chunking architecture."""
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

    sem_dir_224, drv_dir_224 = None, None
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
    if args.stride > 1:
        img_files = img_files[::args.stride]

    chunks = [img_files[i:i + args.chunk_size] for i in range(0, len(img_files), args.chunk_size)]
    processed = 0

    pbar_seq = tqdm(total=len(chunks), desc=f"[{device}] {seq_name} (Chunks)", leave=False)

    for chunk in chunks:
        # Check if entire chunk is already computed
        needs_processing = False
        for img_path in chunk:
            fname = os.path.splitext(os.path.basename(img_path))[0]
            if not os.path.exists(os.path.join(sem_dir, f"{fname}.npy")) or args.force:
                needs_processing = True
                break
                
        if not needs_processing:
            processed += len(chunk)
            pbar_seq.update(1)
            continue

        h, w = None, None
        tmp_vid_dir = tempfile.mkdtemp(prefix="sam2_vid_chunk_")

        try:
            # 1. Symlink images sequentially so SAM 2 can load them natively
            for idx, img_path in enumerate(chunk):
                # The .jpg suffix is REQUIRED because SAM2's misc.load_video_frames_from_jpg_images strictly filters for it
                os.symlink(os.path.abspath(img_path), os.path.join(tmp_vid_dir, f"{idx:05d}.jpg"))

            inference_state = video_predictor.init_state(video_path=tmp_vid_dir)
            
            uid = 0
            uid_to_class = {}
            uid_to_conf = {}

            # 2. Keyframe Object Detection with Grounding DINO
            for local_idx in range(0, len(chunk), args.keyframe_interval):
                img_path = chunk[local_idx]
                image_source, image = load_image(img_path)
                if h is None: 
                    h, w, _ = image_source.shape

                # Ensure tensor format
                boxes, confidences, labels = predict(
                    model=grounding_model,
                    image=image,
                    caption=TEXT_PROMPT,
                    box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold,
                    device=device
                )

                if len(boxes) > 0:
                    boxes = boxes * torch.Tensor([w, h, w, h]).to(boxes.device)
                    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()
                    confs = confidences.cpu().numpy().tolist()

                    for k in range(len(input_boxes)):
                        uid += 1
                        uid_to_class[uid] = labels[k]
                        uid_to_conf[uid] = confs[k]
                        
                        _, _, _ = video_predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=local_idx,
                            obj_id=uid,
                            box=input_boxes[k],
                        )

            # 3. Propagate masks smoothly across the chunk
            # video_predictor yields every frame in the video_path sequentially
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
                img_path = chunk[out_frame_idx]
                fname = os.path.splitext(os.path.basename(img_path))[0]

                masks_list = []
                classes_list = []
                confs_list = []

                for i, out_obj_id in enumerate(out_obj_ids):
                    # Mask logits > 0.0 defines the boolean mask
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze(0) 
                    masks_list.append(mask)
                    classes_list.append(uid_to_class[out_obj_id])
                    confs_list.append(uid_to_conf[out_obj_id])

                label_map, drv_map = masks_to_labelmap(masks_list, classes_list, confs_list, h, w)

                np.save(os.path.join(sem_dir, f"{fname}.npy"), label_map)
                np.save(os.path.join(drv_dir, f"{fname}.npy"), drv_map)

                if sem_dir_224 and drv_dir_224:
                    label_224 = cv2.resize(label_map, (224, 224), interpolation=cv2.INTER_NEAREST)
                    drv_224 = cv2.resize(drv_map, (224, 224), interpolation=cv2.INTER_NEAREST)
                    np.save(os.path.join(sem_dir_224, f"{fname}.npy"), label_224)
                    np.save(os.path.join(drv_dir_224, f"{fname}.npy"), drv_224)

                if args.viz:
                    orig_img = cv2.imread(str(img_path))
                    save_visualisation(None, label_map, os.path.join(sem_vis_dir, f"{fname}_sem.png"), orig_img)
                    save_drivability_visualisation(None, drv_map, os.path.join(drv_vis_dir, f"{fname}_drv.png"), orig_img)

                    if sem_dir_224 and drv_dir_224:
                        img_224 = cv2.resize(orig_img, (224, 224), interpolation=cv2.INTER_AREA)
                        save_visualisation(None, label_224, os.path.join(sem_vis_dir_224, f"{fname}_sem.png"), img_224)
                        save_drivability_visualisation(None, drv_224, os.path.join(drv_vis_dir_224, f"{fname}_drv.png"), img_224)
                
                processed += 1

            # Cleanup SAM 2 cache for this chunk to prevent VRAM overflow
            video_predictor.reset_state(inference_state)

        except Exception as e:
            print(f"  [{device}] Error processing chunk starting at {chunk[0]}: {e}")

        finally:
            shutil.rmtree(tmp_vid_dir, ignore_errors=True)
            torch.cuda.empty_cache()

        pbar_seq.update(1)

    pbar_seq.close()
    return processed


def gpu_worker(gpu_id, seq_paths, args, return_dict):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()

    video_predictor, grounding_model = build_models(
        device, args.sam2_checkpoint, args.sam2_config,
        args.gdino_config, args.gdino_checkpoint
    )

    total = 0
    for seq_path in seq_paths:
        if os.path.isdir(seq_path):
            seq_name = os.path.basename(seq_path)
            out_224 = os.path.join(args.data_root_224, seq_name) if args.data_root_224 else None
            total += process_sequence(seq_path, video_predictor, grounding_model, device, args, out_224)
    
    return_dict[gpu_id] = total


def main():
    parser = argparse.ArgumentParser(description="SAM 2 Video Predictor Preprocessing")
    parser.add_argument("--data_root", type=str, default="/data")
    parser.add_argument("--seq", nargs="*", default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--box_threshold", type=float, default=0.35)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--chunk_size", type=int, default=150, help="Frames per trackable chunk")
    parser.add_argument("--keyframe_interval", type=int, default=5, help="Frequency of Grounding DINO detection")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--data_root_224", type=str, default="/data_224")
    
    parser.add_argument("--sam2_checkpoint", type=str, default="./checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_config", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--gdino_config", type=str, default="grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py") # default to SwinB if available
    parser.add_argument("--gdino_checkpoint", type=str, default="gdino_checkpoints/groundingdino_swinb_cogcoor.pth")

    args = parser.parse_args()

    if not os.path.isdir(args.data_root):
        print(f"Error: {args.data_root} is not a directory")
        sys.exit(1)

    if args.seq:
        seq_folders = [os.path.join(args.data_root, s) for s in args.seq]
    else:
        seq_folders = sorted([
            os.path.join(args.data_root, d)
            for d in os.listdir(args.data_root)
            if os.path.isdir(os.path.join(args.data_root, d))
        ])

    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    
    print("\n[Video Tracking] Preprocess Semantics")
    print(f"  Data root: {args.data_root}")
    print(f"  Sequences: {len(seq_folders)}")
    print(f"  Chunk Size: {args.chunk_size} | Keyframe Interval: {args.keyframe_interval}")
    print(f"  GPUs: {num_gpus} | Visualise: {args.viz} | Force: {args.force}\n")

    t0 = time.time()

    if num_gpus > 1:
        gpu_seqs = [[] for _ in range(num_gpus)]
        for i, seq in enumerate(seq_folders):
            gpu_seqs[i % num_gpus].append(seq)

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
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda"):
            torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
            
        video_predictor, grounding_model = build_models(
            device, args.sam2_checkpoint, args.sam2_config,
            args.gdino_config, args.gdino_checkpoint
        )

        total_processed = 0
        for seq_path in seq_folders:
            if os.path.isdir(seq_path):
                out_224 = os.path.join(args.data_root_224, os.path.basename(seq_path)) if args.data_root_224 else None
                total_processed += process_sequence(seq_path, video_predictor, grounding_model, device, args, out_224)

    elapsed = time.time() - t0
    fps = total_processed / elapsed if elapsed > 0 else 0
    print(f"\nDone! Processed {total_processed} frames in {elapsed:.1f}s ({fps:.1f} fps)")


if __name__ == "__main__":
    main()
