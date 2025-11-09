IMG_INPUT_DIR = Path("robotcycle/img_in")

for img_filename in IMG_INPUT_DIR.iterdir():
    if not img_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        continue  # skip non-image files

    img_path = img_filename.as_posix()
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