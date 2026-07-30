"""
Drawing layer. Turns detection results into annotated images for display.
No model inference or database code here - just cv2 drawing calls.
"""

import cv2

ASSIGNED_COLOR = (0, 165, 255)  # orange (BGR) - slot assigned/reused this run
OCCUPIED_COLOR = (0, 0, 255)    # red (BGR) - already occupied
EMPTY_COLOR = (0, 255, 0)       # green (BGR) - still empty


def draw_slot_status(image_bgr, boxes_info, occupied_class_id, assigned_slot=None):
    """
    Draws every detected slot on top of the parking image:
      - occupied slots: colored rectangle + a big diagonal X through the box
      - the slot assigned/reused this run: same, but highlighted in a
        different color so it's obvious at a glance
      - empty slots: outline only, no X

    Always built from this run's live detections (boxes_info) and the
    outcome of this run's assignment, so it can never drift out of sync
    with the Occupied/Empty counts shown next to it.
    """
    vis = image_bgr.copy()

    for b in boxes_info:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        is_assigned = assigned_slot is not None and b["slot_id"] == assigned_slot
        is_occupied = b["cls"] == occupied_class_id

        if is_assigned:
            color, thickness = ASSIGNED_COLOR, 4
        elif is_occupied:
            color, thickness = OCCUPIED_COLOR, 2
        else:
            color, thickness = EMPTY_COLOR, 2

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        if is_occupied or is_assigned:
            cv2.line(vis, (x1, y1), (x2, y2), color, 3)
            cv2.line(vis, (x2, y1), (x1, y2), color, 3)

        cv2.putText(
            vis, f"Slot {b['slot_id']}", (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

    return vis
