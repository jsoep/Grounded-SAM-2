import os
import cv2

IMG_OUT_DIR = "robotcycle/img_out/"

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

if __name__ == "__main__":
    img_to_video(fps=10, out_name="output_video1.mp4")