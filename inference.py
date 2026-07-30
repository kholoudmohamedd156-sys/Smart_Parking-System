"""
Model inference layer.

Everything here is about turning an image into structured detections.
Nothing in this file touches Streamlit or the database - that separation
is what lets the detection logic be tested/reused independently of the app.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import easyocr

PLATE_MODEL_PATH = r"D:\Smart Parking system\runs\detect\plate_detector-2\weights\best.pt"

# NOTE: "occupancy_detector_v2" (with the _v2 suffix) is NOT this model -
# train_model_parking.ipynb trains a *different*, unrelated dataset
# ('parking-lot-1/data.yaml') under that run name, producing a model whose
# classes are {0:'0',1:'1',2:'2',3:'3',4:'4',5:'object'} - slot/position
# labels, not occupancy status. The real occupancy model is
# "occupancy_detector" (no _v2), confirmed to have {0:'empty',1:'occupied'}.
# Rename the parking-lot-1 training run (e.g. name="slot_position_detector")
# so it never collides with a real run folder like this again.
PARKING_MODEL_PATH = r"D:\Smart Parking system\runs\detect\occupancy_detector\weights\best.pt"


def load_models():
    """Loads both YOLO models and the OCR reader. Call this once and reuse."""
    plate_model = YOLO(PLATE_MODEL_PATH)
    parking_model = YOLO(PARKING_MODEL_PATH)
    ocr_reader = easyocr.Reader(["en"])
    return plate_model, parking_model, ocr_reader


def resolve_slot_class_ids(parking_model):
    """
    Reads the empty/occupied class ids from the model's own label map
    instead of hardcoding them, so re-exporting/re-training the model with
    a different class order can't silently break the occupied/empty counts.

    The real occupancy model has both a generic pair ('empty'/'occupied')
    and a per-slot pair ('space-empty'/'space-occupied'). We prefer the
    'space-*' pair when it's present, since those are the classes drawn as
    boxes around individual parking spots - 'empty'/'occupied' alone (without
    the "space-" prefix) is ambiguous with the model's other classes
    ('car', 'motorcycle') and shouldn't be assumed to mean the same thing.
    """
    names = {idx: name.lower() for idx, name in parking_model.names.items()}

    def find(*keywords):
        return next((idx for idx, label in names.items()
                     if any(k in label for k in keywords)), None)

    empty_id = find("space-empty") or find("space_empty")
    occupied_id = find("space-occupied") or find("space_occupied")

    if empty_id is None:
        empty_id = find("empty", "vacant", "free")
    if occupied_id is None:
        occupied_id = find("occupied", "busy", "taken")

    if empty_id is None or occupied_id is None:
        all_numeric = all(name.isdigit() for name in names.values() if name != "object")
        hint = (
            "\nThese look like slot/position labels rather than occupancy "
            "labels - this is almost certainly the wrong weights file. "
            "train_model_parking.ipynb trains on 'parking-lot-1/data.yaml' "
            "under the run name 'occupancy_detector_v2', which silently "
            "overwrites/creates a folder with that name that has nothing to "
            "do with empty/occupied detection. Point PARKING_MODEL_PATH at "
            "the best.pt from training on the real occupancy dataset "
            "instead (the one with car/empty/motorcycle/occupied/"
            "space-empty/space-occupied classes)."
            if all_numeric else ""
        )
        raise RuntimeError(
            f"Could not resolve 'empty'/'occupied' classes from the parking "
            f"model's class names: {parking_model.names}.{hint}\n"
            f"If this really is the right model, update the keyword lists "
            f"in resolve_slot_class_ids() to match the real label names."
        )
    return empty_id, occupied_id


def load_image_bgr(uploaded_file):
    """
    Decode a Streamlit upload the same way the models were trained/validated
    (cv2-style, 3-channel BGR).

    PIL.Image.open() + np.array() (the old approach) returns RGB - and RGBA
    for PNGs - which is a different channel order than the BGR data the
    models were trained and tested on. Feeding a BGR-trained model an RGB
    array silently degrades/flips its predictions instead of erroring out,
    which is what caused the occupied/empty miscounts. cv2.imdecode keeps
    inference consistent with training and always drops any alpha channel.
    """
    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def detect_parking_slots(parking_model, image_bgr, empty_class_id, occupied_class_id, conf):
    """
    Runs the occupancy model on a parking-lot image.

    Returns:
        boxes_info: list of {"x1","y1","x2","y2","cls","slot_id"} dicts,
            one per detected slot, numbered top-to-bottom then
            left-to-right (stable as long as the camera angle is fixed).
        occupied_count, empty_count: raw counts for this specific image,
            straight from the model - matches what's on screen exactly.
    """
    results = parking_model.predict(image_bgr, conf=conf)

    boxes_info = []
    occupied_count = 0
    empty_count = 0

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])

        # Only the slot classes matter here - the model also detects
        # "car"/"motorcycle"/etc as separate boxes on the same cars, and
        # counting those too would corrupt both the counts and the slot
        # numbering.
        if cls not in (empty_class_id, occupied_class_id):
            continue

        boxes_info.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cls": cls})
        if cls == occupied_class_id:
            occupied_count += 1
        else:
            empty_count += 1

    boxes_info.sort(key=lambda b: (round(b["y1"] / 50), b["x1"]))
    for i, b in enumerate(boxes_info):
        b["slot_id"] = i + 1

    return boxes_info, occupied_count, empty_count


def detect_plate(plate_model, ocr_reader, image_bgr, conf):
    """
    Runs the plate detector + OCR on a gate-camera image.
    Returns (plate_text, annotated_image_bgr). plate_text is "Not Detected"
    if no plate box was found or OCR couldn't read the crop.

    plate_text is normalized (uppercased, extra whitespace collapsed)
    because EasyOCR's casing isn't consistent run-to-run - the same
    physical plate can come back as "DL7CQ1939" one time and "dl7cq1939"
    (or a mix) another. assign_or_reuse_slot() matches plates by exact
    string equality, so inconsistent casing would make the same car look
    like a different vehicle on a repeat scan and defeat the
    already-parked/reuse check entirely.
    """
    results = plate_model.predict(image_bgr, conf=conf)
    plate_text = "Not Detected"

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = image_bgr[y1:y2, x1:x2]
        ocr_result = ocr_reader.readtext(crop)
        if ocr_result:
            plate_text = " ".join(r[1] for r in ocr_result)
            plate_text = " ".join(plate_text.upper().split())

    annotated_image_bgr = results[0].plot()
    return plate_text, annotated_image_bgr
