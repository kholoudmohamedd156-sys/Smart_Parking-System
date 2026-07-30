"""
main.py
Streamlit front-end for the Smart Parking System.

Pages:
  - Check-In   : upload/capture an image, detect the plate with YOLO, read it with EasyOCR,
                 confirm, and log the vehicle as parked.
  - Check-Out  : same detection flow (or manual entry) to close out a parked vehicle.
  - Dashboard  : live occupancy metrics + tables of active vehicles and full history.
  - Settings   : configure total number of spots.

Run with:  streamlit run main.py
"""

import os
import cv2
import numpy as np
import streamlit as st
import pandas as pd

import database as db

# ----------------------------------------------------------------------------
# Page config & one-time setup
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Smart Parking System", page_icon="🅿️", layout="wide")
db.init_db()

MODEL_PATH = "yolov8n.pt"          # swap for "yolo26n.pt" / a plate-specific model if you have one
CONF_THRESHOLD = 0.5
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


# ----------------------------------------------------------------------------
# Cached model loaders (loaded once per session, not on every rerun)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO model...")
def load_yolo_model():
    from ultralytics import YOLO
    return YOLO(MODEL_PATH)


@st.cache_resource(show_spinner="Loading OCR engine...")
def load_ocr_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=False)


def detect_and_read_plate(image_bgr: np.ndarray):
    """
    Runs YOLO on the image to find the plate/vehicle box, crops it, and runs
    EasyOCR on the crop. Returns (plate_text, annotated_image, crop) or
    (None, annotated_image, None) if nothing usable was found.
    """
    model = load_yolo_model()
    reader = load_ocr_reader()

    results = model.predict(image_bgr, conf=CONF_THRESHOLD, verbose=False)
    annotated = image_bgr.copy()

    if not results or len(results[0].boxes) == 0:
        return None, annotated, None

    # Take the highest-confidence box as the plate/vehicle region
    boxes = results[0].boxes
    best_idx = int(boxes.conf.argmax())
    x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx])
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None, annotated, None

    ocr_result = reader.readtext(crop)
    if not ocr_result:
        return None, annotated, crop

    plate_text = " ".join([r[1] for r in ocr_result]).strip().upper()
    return plate_text, annotated, crop


def image_uploader_to_bgr(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.title("🅿️ Smart Parking")
page = st.sidebar.radio("Navigate", ["Dashboard", "Check-In", "Check-Out", "Settings"])

available = db.get_available_spots()
total = db.get_total_spots()
st.sidebar.metric("Available spots", f"{available} / {total}")

# ----------------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------------
if page == "Dashboard":
    st.title("Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total spots", total)
    col2.metric("Occupied", total - available)
    col3.metric("Available", available)

    st.subheader("Currently parked")
    active = db.get_active_vehicles()
    if active:
        df_active = pd.DataFrame([dict(r) for r in active])
        st.dataframe(df_active[["plate_number", "entry_time"]], use_container_width=True)
    else:
        st.info("No vehicles currently parked.")

    st.subheader("Recent history")
    logs = db.get_all_logs()
    if logs:
        df_logs = pd.DataFrame([dict(r) for r in logs])
        st.dataframe(
            df_logs[["plate_number", "entry_time", "exit_time", "status"]],
            use_container_width=True,
        )
    else:
        st.info("No history yet.")

# ----------------------------------------------------------------------------
# Check-In
# ----------------------------------------------------------------------------
elif page == "Check-In":
    st.title("Vehicle Check-In")

    if db.get_available_spots() <= 0:
        st.error("The lot is full — no available spots.")

    source = st.radio("Image source", ["Upload", "Camera"], horizontal=True)
    image_file = (
        st.file_uploader("Upload a photo of the vehicle/plate", type=["jpg", "jpeg", "png"])
        if source == "Upload"
        else st.camera_input("Take a photo")
    )

    if image_file is not None:
        img_bgr = image_uploader_to_bgr(image_file)
        with st.spinner("Detecting plate..."):
            plate_text, annotated, crop = detect_and_read_plate(img_bgr)

        col1, col2 = st.columns(2)
        col1.image(annotated[:, :, ::-1], caption="Detection result", use_container_width=True)
        if crop is not None:
            col2.image(crop[:, :, ::-1], caption="Plate crop", use_container_width=True)

        plate_number = st.text_input(
            "Detected plate number (edit if needed)", value=plate_text or ""
        )

        if st.button("Confirm Check-In", type="primary", disabled=not plate_number.strip()):
            snapshot_path = os.path.join(SNAPSHOT_DIR, f"{plate_number}_in.jpg")
            cv2.imwrite(snapshot_path, img_bgr)
            ok = db.check_in(plate_number.strip().upper(), image_path=snapshot_path)
            if ok:
                st.success(f"'{plate_number}' checked in successfully.")
            else:
                st.error("Check-in failed: plate already parked, or lot is full.")

# ----------------------------------------------------------------------------
# Check-Out
# ----------------------------------------------------------------------------
elif page == "Check-Out":
    st.title("Vehicle Check-Out")

    method = st.radio("Identify vehicle by", ["Manual plate entry", "Upload photo"], horizontal=True)

    if method == "Manual plate entry":
        plate_number = st.text_input("Plate number").strip().upper()
        if st.button("Check-Out", type="primary", disabled=not plate_number):
            ok = db.check_out(plate_number)
            if ok:
                st.success(f"'{plate_number}' checked out successfully.")
            else:
                st.error("No matching parked vehicle found for that plate.")

    else:
        image_file = st.file_uploader("Upload a photo of the vehicle/plate", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            img_bgr = image_uploader_to_bgr(image_file)
            with st.spinner("Detecting plate..."):
                plate_text, annotated, crop = detect_and_read_plate(img_bgr)

            st.image(annotated[:, :, ::-1], caption="Detection result", use_container_width=True)
            plate_number = st.text_input(
                "Detected plate number (edit if needed)", value=plate_text or ""
            )
            if st.button("Confirm Check-Out", type="primary", disabled=not plate_number.strip()):
                ok = db.check_out(plate_number.strip().upper())
                if ok:
                    st.success(f"'{plate_number}' checked out successfully.")
                else:
                    st.error("No matching parked vehicle found for that plate.")

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
elif page == "Settings":
    st.title("Settings")
    current_total = db.get_total_spots()
    new_total = st.number_input(
        "Total parking spots", min_value=1, value=current_total, step=1
    )
    if st.button("Save", type="primary"):
        db.set_total_spots(int(new_total))
        st.success(f"Total spots updated to {int(new_total)}.")
        st.rerun()