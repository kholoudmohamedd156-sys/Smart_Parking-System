import sys
import os

# database.py lives in a sibling folder (../database), not next to this
# file, so it isn't found by a plain "import database" - this adds that
# folder to Python's search path first. This is based on THIS file's own
# location, so it works no matter what folder you run streamlit from.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database"))

import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

from database import (
    create_database,
    insert_data,
    get_all_data,
    sync_slots,
    assign_next_empty_slot,
    is_plate_parked,
    find_car_by_plate,
    get_all_slots,
)

# ==============================================================================
# MODELS + DETECTION
# ==============================================================================

PLATE_MODEL_PATH = r"D:\Smart Parking system\runs\detect\plate_detector-2\weights\best.pt"
PARKING_MODEL_PATH = r"D:\Smart Parking system\runs\detect\occupancy_detector_v2\weights\best.pt"

EMPTY_CLASS_ID = 1      # 'empty' - well represented in training (mAP50 0.972)
OCCUPIED_CLASS_ID = 3   # 'occupied' - well represented in training (mAP50 0.901)

plate_model = YOLO(PLATE_MODEL_PATH)
parking_model = YOLO(PARKING_MODEL_PATH)
reader = easyocr.Reader(['en'])


def detect_parking_slots(img, conf_threshold=0.5):
    """
    Runs the occupancy model on the parking-lot image and returns each
    detected slot's position, class, and a stable slot number.
    Only "empty" / "occupied" classes are kept - "car" / "motorcycle"
    boxes on the same cars would mess up the slot numbering if counted.
    """
    results = parking_model.predict(img, conf=conf_threshold)

    boxes_info = []
    occupied = 0
    empty = 0

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])

        if cls not in (EMPTY_CLASS_ID, OCCUPIED_CLASS_ID):
            continue

        boxes_info.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cls": cls})

        if cls == OCCUPIED_CLASS_ID:
            occupied += 1
        elif cls == EMPTY_CLASS_ID:
            empty += 1

    # Number each slot by position (top-to-bottom, then left-to-right).
    boxes_info.sort(key=lambda b: (round(b["y1"] / 50), b["x1"]))
    for i, b in enumerate(boxes_info):
        b["slot_id"] = i + 1

    return boxes_info, occupied, empty


def detect_plate(img, conf_threshold=0.5):
    """Runs the plate model, OCRs whichever plate box(es) it finds."""
    results = plate_model.predict(img, conf=conf_threshold)
    plate_number = "Not Detected"

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = img[y1:y2, x1:x2]
        ocr = reader.readtext(crop)
        if ocr:
            plate_number = " ".join(r[1] for r in ocr).upper()

    plotted_img = results[0].plot()
    return plate_number, plotted_img


# ==============================================================================
# STREAMLIT APP
# ==============================================================================

create_database()

st.set_page_config(page_title="Smart Parking System", page_icon="🚧", layout="wide")

# ------------------------------------------------------------------------------
# Theme - boom-gate / control-room look: dark asphalt background, hazard-amber
# accent, condensed display type for headers and a monospace "LED readout"
# treatment for the plate number and slot badges.
# ------------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@600;700&display=swap');

:root {
    --sky:        #EEF8FE;   /* page background */
    --sky-dk:     #DEF0FB;   /* recessed / header surfaces */
    --paper:      #FFFFFF;   /* card / ticket background */
    --border:     #CBE7F8;   /* hairline borders */
    --ink:        #1F3B4D;   /* headings, primary text */
    --slate:      #6C8FA3;   /* secondary text */
    --babyblue:   #1E4C8C;   /* primary accent - deep navy blue */
    --babyblue-dk:#123563;   /* accent hover/active - darker still */
    --babyblue-lt:#D9E6F5;   /* light accent fills */
    --sage:       #3F9E92;   /* success */
    --sage-bg:    #DFF4F1;
    --brick:      #DD6255;   /* denied/error */
    --brick-bg:   #FCE7E4;
    --amber:      #D69A2D;   /* warning */
    --amber-bg:   #FBF0DB;
    --shadow-sm:  0 1px 3px rgba(31,59,77,0.07), 0 1px 1px rgba(31,59,77,0.04);
    --shadow-md:  0 14px 30px rgba(31,59,77,0.11), 0 3px 8px rgba(31,59,77,0.06);
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; font-size: 1.05rem; }

.stApp {
    background: radial-gradient(circle at 15% -10%, #FFFFFF 0%, var(--sky) 45%);
    color: var(--ink);
}

section[data-testid="stSidebar"] { display: none; }

/* Full-width layout with generous side padding instead of a centered column */
div.block-container {
    max-width: 100%;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
    padding-left: 4rem;
    padding-right: 4rem;
}

/* Ticket-strip divider - signature element, used between sections */
.barrier {
    height: 12px;
    border-radius: 999px;
    background: repeating-linear-gradient(135deg, var(--babyblue) 0px, var(--babyblue) 18px, var(--paper) 18px, var(--paper) 36px);
    box-shadow: var(--shadow-sm);
    margin: 10px 0 36px 0;
    opacity: 0.95;
}

.app-header { display: flex; align-items: center; gap: 18px; margin-bottom: 6px; }
.app-badge {
    display: flex; align-items: center; justify-content: center;
    width: 68px; height: 68px; flex-shrink: 0;
    background: var(--babyblue);
    color: var(--paper);
    border-radius: 18px;
    font-size: 2rem;
    box-shadow: var(--shadow-md);
}
.app-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 3rem;
    letter-spacing: -0.01em;
    color: var(--ink);
    margin: 0;
}
.app-sub {
    color: var(--slate);
    font-size: 1.2rem;
    margin: 10px 0 40px 0;
    line-height: 1.55;
}

.section-label {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 1.65rem;
    color: var(--ink);
    margin: 0 0 18px 0;
    padding-left: 14px;
    border-left: 4px solid var(--babyblue);
    display: flex;
    align-items: center;
    gap: 10px;
}

/* File uploader cards */
section[data-testid="stFileUploaderDropzone"] {
    background: var(--paper);
    border: 2px dashed var(--border);
    border-radius: 16px;
    transition: border-color 0.15s ease;
}
section[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--babyblue); }

/* Buttons */
div.stButton > button {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    background: var(--babyblue);
    color: var(--paper);
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.9rem;
    box-shadow: var(--shadow-sm);
    transition: transform 0.12s ease, box-shadow 0.12s ease, background 0.12s ease;
}
div.stButton > button:hover {
    background: var(--babyblue-dk);
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}
div.stButton > button:active { transform: translateY(0); }

/* Text input (Where Is My Car search box) */
div[data-testid="stTextInput"] input {
    background: var(--paper);
    border: 2px solid var(--border);
    border-radius: 12px;
    color: var(--ink);
    font-size: 1.05rem;
    padding: 0.7rem 1.1rem;
}
div[data-testid="stTextInput"] input:focus {
    border-color: var(--babyblue);
    box-shadow: 0 0 0 4px var(--babyblue-lt);
}

/* Plate readout - printed ticket-stub styling with a torn top edge */
.plate-readout {
    position: relative;
    display: inline-block;
    background: var(--paper);
    color: var(--ink);
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 2.1rem;
    letter-spacing: 0.1em;
    padding: 20px 34px 16px 34px;
    border-radius: 6px 6px 14px 14px;
    border: 1.5px solid var(--border);
    box-shadow: var(--shadow-md);
    margin: 10px 0 26px 0;
}
.plate-readout::before {
    content: "";
    position: absolute;
    top: -7px; left: 0; right: 0;
    height: 14px;
    background-image: radial-gradient(circle 7px, var(--sky) 7px, transparent 8px);
    background-size: 24px 14px;
    background-position: -2px 0;
    background-repeat: repeat-x;
}

/* Slot badge */
.slot-badge {
    display: inline-block;
    background: var(--babyblue);
    color: var(--paper);
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2rem;
    padding: 12px 30px;
    border-radius: 12px;
    box-shadow: var(--shadow-md);
    margin: 10px 0 26px 0;
}

/* Status banners */
.banner {
    border-radius: 14px;
    padding: 18px 22px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 1.08rem;
    box-shadow: var(--shadow-sm);
    margin: 14px 0 26px 0;
}
.banner-ok      { background: var(--sage-bg);  color: var(--sage); }
.banner-warn    { background: var(--amber-bg); color: var(--amber); }
.banner-denied  { background: var(--brick-bg); color: var(--brick); }

/* Status pill table (Slots Status / History) */
.pill-table { width: 100%; border-collapse: collapse; }
.pill-table th {
    text-align: left;
    font-family: 'Inter', sans-serif;
    font-size: 0.86rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--slate);
    padding: 16px 20px;
    background: var(--sky-dk);
}
.pill-table th:first-child { border-radius: 14px 0 0 0; }
.pill-table th:last-child  { border-radius: 0 14px 0 0; }
.pill-table td {
    padding: 16px 20px;
    border-bottom: 1.5px solid var(--border);
    font-size: 1.05rem;
    color: var(--ink);
}
.pill-table tr:last-child td { border-bottom: none; }
.pill-table tr:hover td { background: var(--sky); }

.pill { display: inline-block; padding: 5px 16px; border-radius: 999px; font-size: 0.9rem; font-weight: 600; }
.pill-empty    { background: var(--sage-bg);  color: var(--sage); }
.pill-occupied { background: var(--brick-bg); color: var(--brick); }

.card {
    background: var(--paper);
    border: 1.5px solid var(--border);
    border-radius: 18px;
    padding: 6px;
    margin-bottom: 18px;
    box-shadow: var(--shadow-sm);
    overflow: hidden;
}
.card .pill-table { padding: 0 4px; }

.team-pill {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: var(--paper);
    border: 1.5px solid var(--border);
    border-radius: 999px;
    padding: 10px 22px 10px 12px;
    margin: 6px 12px 6px 0;
    font-size: 1.05rem;
    text-decoration: none !important;
    color: var(--ink) !important;
    box-shadow: var(--shadow-sm);
    transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease;
}
.team-pill img { border-radius: 999px; }
.team-pill:hover { border-color: var(--babyblue); transform: translateY(-2px); box-shadow: var(--shadow-md); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="barrier"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-header"><div class="app-badge">🚧</div>'
    '<p class="app-title">Smart Parking System</p></div>',
    unsafe_allow_html=True,
)
st.markdown('<p class="app-sub">Scan the gate camera and the lot camera — the system reads the plate, checks it in, and assigns a spot.</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="section-label">📷 Gate Camera</p>', unsafe_allow_html=True)
    gate_file = st.file_uploader("Upload Gate Image", type=["jpg", "jpeg", "png"], key="gate", label_visibility="collapsed")

with col2:
    st.markdown('<p class="section-label">🅿️ Parking Camera</p>', unsafe_allow_html=True)
    parking_file = st.file_uploader("Upload Parking Image", type=["jpg", "jpeg", "png"], key="parking", label_visibility="collapsed")

st.write("")
analyze_clicked = st.button("Analyze")

if analyze_clicked:

    plate_number = "Not Detected"
    occupied = 0
    empty = 0
    boxes_info = []
    img2 = None

    # 1) Analyze the parking image first, so we know how many spots the lot
    #    actually has, and make sure the slots table has a row for each one.
    if parking_file is not None:
        image2 = Image.open(parking_file)
        img2 = np.array(image2)

        boxes_info, occupied, empty = detect_parking_slots(img2, conf_threshold=0.5)

        total_slots = len(boxes_info)
        sync_slots(total_slots)
        # (occupied/empty are still computed above - used below for the
        # slot assignment and saved to the database - just not displayed
        # as their own "Parking Status" section.)

    # 2) Now analyze the gate image, and reserve a *specific* slot instead
    #    of just incrementing/decrementing a counter.
    assigned_slot = None
    entry_granted = False

    if gate_file is not None:
        image = Image.open(gate_file)
        img = np.array(image)

        plate_number, plotted = detect_plate(img, conf_threshold=0.5)

        st.markdown('<p class="section-label">Plate Number</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="plate-readout">{plate_number}</div>', unsafe_allow_html=True)
        st.image(plotted, caption="Detected Plate", width=420)

        entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        already_parked_slot = is_plate_parked(plate_number)
        if plate_number == "Not Detected":
            assigned_slot = None
        elif already_parked_slot is not None:
            assigned_slot = already_parked_slot
            st.markdown(
                f'<div class="banner banner-denied">⛔ &nbsp;\'{plate_number}\' is already parked '
                f'in Slot #{already_parked_slot} — entry denied.</div>',
                unsafe_allow_html=True,
            )
        else:
            assigned_slot = assign_next_empty_slot(plate_number, entry_time)

            if assigned_slot is not None:
                entry_granted = True
                st.markdown(f'<div class="slot-badge">🅿️ Slot #{assigned_slot}</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="banner banner-ok">✅ &nbsp;Entry granted — this slot has been assigned to the car.</div>',
                    unsafe_allow_html=True,
                )

                if boxes_info and img2 is not None:
                    highlight_img = img2.copy()
                    for b in boxes_info:
                        if b["slot_id"] == assigned_slot:
                            color = (0, 165, 255)   # orange - the newly assigned slot
                            thickness = 4
                        elif b["cls"] == EMPTY_CLASS_ID:
                            color = (0, 255, 0)     # green - still empty
                            thickness = 2
                        else:
                            color = (0, 0, 255)     # red - already occupied
                            thickness = 2

                        cv2.rectangle(highlight_img, (b["x1"], b["y1"]), (b["x2"], b["y2"]), color, thickness)
                        cv2.putText(
                            highlight_img, f"Slot {b['slot_id']}", (b["x1"], max(b["y1"] - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                        )
                    st.image(highlight_img, caption=f"Slot #{assigned_slot} assigned (orange)", width=520)
            else:
                st.markdown(
                    '<div class="banner banner-warn">⚠️ &nbsp;Parking lot is full — no empty slots available.</div>',
                    unsafe_allow_html=True,
                )

    if entry_granted:
        insert_data(
            plate_number,
            assigned_slot,
            occupied,
            empty,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        st.markdown('<div class="banner banner-ok">💾 &nbsp;Saved to database.</div>', unsafe_allow_html=True)


def render_pill_table(headers, rows):
    """rows: list of tuples matching headers, where any 'status' cell is
    pre-rendered as an <span class="pill ..."> string."""
    html = ['<table class="pill-table"><thead><tr>']
    for h in headers:
        html.append(f"<th>{h}</th>")
    html.append("</tr></thead><tbody>")
    for row in rows:
        html.append("<tr>")
        for cell in row:
            html.append(f"<td>{cell}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    return "".join(html)


# ------------------------------------------------------------------------------
# Where is my car? - look up a plate and show which slot it's currently in.
# ------------------------------------------------------------------------------
st.markdown('<div class="barrier"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">🔎 Where Is My Car?</p>', unsafe_allow_html=True)

search_col1, search_col2 = st.columns([3, 1])
with search_col1:
    search_plate = st.text_input(
        "Enter plate number",
        key="search_plate",
        label_visibility="collapsed",
        placeholder="Enter your plate number, e.g. ABC 1234",
    )
with search_col2:
    search_clicked = st.button("Search")

if search_clicked:
    if not search_plate.strip():
        st.markdown('<div class="banner banner-warn">⚠️ &nbsp;Enter a plate number to search.</div>', unsafe_allow_html=True)
    else:
        result = find_car_by_plate(search_plate.strip())
        if result is None:
            st.markdown(
                f'<div class="banner banner-denied">⛔ &nbsp;No car with plate \'{search_plate.upper()}\' is currently parked.</div>',
                unsafe_allow_html=True,
            )
        else:
            found_slot, found_entry_time = result
            st.markdown(f'<div class="slot-badge">🅿️ Slot #{found_slot}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="banner banner-ok">✅ &nbsp;Found — parked in Slot #{found_slot} since {found_entry_time}.</div>',
                unsafe_allow_html=True,
            )


st.markdown('<div class="barrier"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">🅿️ Slots Status</p>', unsafe_allow_html=True)
if st.button("Show Slots"):
    slots = get_all_slots()
    if slots:
        rows = []
        for slot_id, status, plate, entry_time in slots:
            pill_class = "pill-empty" if status == "empty" else "pill-occupied"
            pill = f'<span class="pill {pill_class}">{status.title()}</span>'
            rows.append((f"#{slot_id}", pill, plate or "—", entry_time or "—"))
        st.markdown(
            f'<div class="card">{render_pill_table(["Slot", "Status", "Plate", "Entry Time"], rows)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.write("No slots recorded yet. Analyze a parking image first.")


st.markdown('<div class="barrier"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">📋 History</p>', unsafe_allow_html=True)
if st.button("Show All Records"):
    records = get_all_data()
    if records:
        # id, plate, slot, occupied, empty, time - occupied/empty kept in the
        # database for reference but not shown here, the slot number is what
        # actually matters to someone reading their parking history.
        rows = [
            (r[0], r[1], f"#{r[2]}" if r[2] is not None else "—", r[5])
            for r in records
        ]
        st.markdown(
            f'<div class="card">{render_pill_table(["ID", "Plate", "Slot", "Time"], rows)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.write("No records yet.")


st.markdown('<div class="barrier"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-label">👩‍💻 Meet Our Team</p>', unsafe_allow_html=True)

st.markdown("""
<div>

<a class="team-pill" href="https://www.linkedin.com/in/kholouddmohamed" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="18">Kholoud Mohamed
</a>

<a class="team-pill" href="https://www.linkedin.com/in/maryam-hassan-8a9b4a414" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="18">Maryam Hassan
</a>

<a class="team-pill" href="https://eg.linkedin.com/in/mennaallahkhaled" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="18">Mennaallah Khaled
</a>

<a class="team-pill" href="https://www.linkedin.com/in/rawan-mohamed5" target="_blank">
<img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="18">Rawan Mohamed
</a>

</div>
""", unsafe_allow_html=True)