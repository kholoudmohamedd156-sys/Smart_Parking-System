# Smart Parking System using YOLOv8 & EasyOCR

## Project Overview
Modern urban parking facilities often suffer from manual entry tracking, inefficient slot management, and human error. This project presents an automated **Smart Parking & License Plate Recognition System** leveraging YOLOv8 object detection, EasyOCR, SQLite data persistence, and a Streamlit dashboard[cite: 1, 2, 5].

The system performs dual real-time tasks: detecting available vs. occupied parking slots dynamically and extracting vehicle license plate numbers at entry gates to automate space allocation and tracking.

## Objectives
* **Automate Slot Occupancy:** Detect empty and occupied parking spaces dynamically from lot camera feeds[cite: 1, 5].
* **Automate Plate Recognition:** Detect vehicle license plates and extract registration text using OCR[cite: 1, 5].
* **Dynamic Database Management:** Automatically map parking capacity, log entry/exit transactions, and enforce slot assignment rules using SQLite.
* **User-Friendly Dashboard:** Provide a real-time Streamlit interface for lot monitoring, plate lookups ("Where Is My Car?"), and historical logging.

## Core Modules & Architecture

### 1. Computer Vision Pipeline
* **Plate Detection & OCR:** Custom YOLOv8 model detects license plate bounding boxes, which are cropped and passed into EasyOCR for character recognition[cite: 1, 3, 5, 6].
* **Occupancy Detection:** Classifies slots as empty or occupied, spatial sorting (top-to-bottom, left-to-right) ensures stable slot numbering.

### 2. Database Integration (`database.py`)
* Thread-safe SQLite management handling `parking` and `slots` tables[cite: 2].
* **Core Functions:** `sync_slots()`, `assign_next_empty_slot()`, `is_plate_parked()`, and `find_car_by_plate()`[cite: 2].

---

## Dataset
The dataset contains annotated parking lot images and license plate crop regions designed for training and validating detection and OCR performance.

### Typical Classes & Annotations:
* `empty_slot`
* `occupied_slot`
* `license_plate`

### Dataset Structure
```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Model
This project utilizes **YOLOv8** (Ultralytics) for visual object detection and **EasyOCR** for textual recognition.

## Why YOLOv8?
* Fast real-time inference speed
* High detection accuracy on localized targets
* Lightweight architecture (`yolov8n.pt`) compatible with edge GPU deployment

## Evaluation Metrics
The license plate detection model achieved the following metrics:

| Metric | Value |
| :--- | :--- |
| **Precision** | High Convergence |
| **Recall** | High Convergence |
| **mAP0.5** | **> 0.99** |
| **mAP0.5:0.95** | **~ 0.67** |

### Dashboard Capabilities
* **Gate & Lot Analysis:** Simultaneous analysis of entry gate plate capture and overall lot occupancy status[cite: 5].
* **Visual Slot Highlights:** Bounding box visualization (Green = Empty, Red = Occupied, Orange = Newly Assigned)[cite: 5].
* **Where Is My Car?:** Real-time lookup query to find parked vehicles by plate number[cite: 5].
* **Logs & History:** Tables displaying active parking slots and archived entry/exit history[cite: 5].

## Technologies Used
* **Python**
* **YOLOv8 (Ultralytics)**
* **EasyOCR**
* **OpenCV**
* **SQLite3**
* **Streamlit**
* **PyTorch / CUDA**

---

## Future Work
* Support live RTSP video stream feeds for continuous camera processing.
* Deploy hardware triggers (e.g., Raspberry Pi gate barrier integration).
* Add automated parking fee calculation based on duration.
* Build a mobile application for drivers to reserve slots in advance.
* Add defect and unauthorized vehicle parking alert systems.

---

## Team Members
* Kholoud Mohamed
* Maryam Adel
* Menna Khaled
* Rawan Mohamed

## video
<!-- Failed to upload "Smart_Parking_System(NTI).mp4" -->
