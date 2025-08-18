from ultralytics import YOLO

def detect(model, frame):
    """
    Crop khuon mat
    """
    results = model.predict(frame, conf=0.4, iou=0.5)
    boxes = results[0].boxes
    bounding_box_list = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        bounding_box_list.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes, bounding_box_list