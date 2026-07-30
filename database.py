"""
database.py
All sqlite access lives here, in its own "database" folder. parking.db is
created right next to this file (not wherever streamlit happens to be run
from), so it doesn't matter what your working directory is when you run
the app.
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "parking.db")


def _connect():
    return sqlite3.connect(DB_PATH)


def create_database():
    conn = _connect()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS parking (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate TEXT,
        slot INTEGER,
        occupied INTEGER,
        empty INTEGER,
        time TEXT
    )
    """)

    # Migration: if "parking" already existed from before the "slot" column
    # was added, patch it in place so older databases keep working instead
    # of crashing on the INSERT below.
    cursor.execute("PRAGMA table_info(parking)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    if "slot" not in existing_cols:
        cursor.execute("ALTER TABLE parking ADD COLUMN slot INTEGER")

    # Slots table: each row = one specific parking spot, with its current status
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS slots (
        slot_id INTEGER PRIMARY KEY,
        status TEXT DEFAULT 'empty',   -- 'empty' or 'occupied'
        plate TEXT,
        entry_time TEXT
    )
    """)

    conn.commit()
    conn.close()


def sync_slots(num_slots):
    """
    Makes sure the number of rows in the slots table matches the number of
    parking spots the model detected in the latest image.
    If the detected number of spots increased (e.g. first run), new rows are
    added with status 'empty'. Existing slots are never touched here, so an
    occupied slot's status is never overwritten by mistake.
    """
    conn = _connect()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM slots")
    existing = cursor.fetchone()[0]

    if existing < num_slots:
        for slot_id in range(existing + 1, num_slots + 1):
            cursor.execute(
                "INSERT OR IGNORE INTO slots (slot_id, status) VALUES (?, 'empty')",
                (slot_id,)
            )

    conn.commit()
    conn.close()


def assign_next_empty_slot(plate, entry_time):
    """
    Finds the first empty slot (smallest slot_id with status 'empty') and
    reserves it for the newly arrived car.
    Returns the slot_id that was assigned, or None if the lot is full.
    """
    conn = _connect()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT slot_id FROM slots WHERE status = 'empty' ORDER BY slot_id ASC LIMIT 1"
    )
    row = cursor.fetchone()

    if row is None:
        conn.close()
        return None

    slot_id = row[0]

    cursor.execute(
        "UPDATE slots SET status = 'occupied', plate = ?, entry_time = ? WHERE slot_id = ?",
        (plate, entry_time, slot_id)
    )

    conn.commit()
    conn.close()
    return slot_id


def is_plate_parked(plate):
    """
    True if this exact plate currently occupies a slot. Used to stop the
    same car from being checked in twice while it's still parked.
    """
    conn = _connect()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT slot_id FROM slots WHERE plate = ? AND status = 'occupied'",
        (plate,)
    )
    row = cursor.fetchone()

    conn.close()
    return row[0] if row else None


def find_car_by_plate(plate):
    """
    "Where is my car?" lookup: returns (slot_id, entry_time) for the slot
    this plate currently occupies, or None if it isn't parked right now.
    Matching is case-insensitive since OCR casing can vary run to run.
    """
    conn = _connect()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT slot_id, entry_time FROM slots WHERE UPPER(plate) = UPPER(?) AND status = 'occupied'",
        (plate,)
    )
    row = cursor.fetchone()

    conn.close()
    return row  # (slot_id, entry_time) or None


def free_slot_by_plate(plate):
    """
    Frees up the slot occupied by a given plate (e.g. when the car leaves).
    Returns the freed slot_id, or None if the plate wasn't found in any slot.
    """
    conn = _connect()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT slot_id FROM slots WHERE plate = ? AND status = 'occupied'",
        (plate,)
    )
    row = cursor.fetchone()

    if row is None:
        conn.close()
        return None

    slot_id = row[0]

    cursor.execute(
        "UPDATE slots SET status = 'empty', plate = NULL, entry_time = NULL WHERE slot_id = ?",
        (slot_id,)
    )

    conn.commit()
    conn.close()
    return slot_id


def get_all_slots():
    conn = _connect()
    cursor = conn.cursor()

    cursor.execute("SELECT slot_id, status, plate, entry_time FROM slots ORDER BY slot_id ASC")
    rows = cursor.fetchall()

    conn.close()
    return rows


def insert_data(plate, slot, occupied, empty, time):
    conn = _connect()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO parking (plate, slot, occupied, empty, time)
    VALUES (?, ?, ?, ?, ?)
    """, (plate, slot, occupied, empty, time))

    conn.commit()
    conn.close()


def get_all_data():
    conn = _connect()
    cursor = conn.cursor()

    cursor.execute("SELECT id, plate, slot, occupied, empty, time FROM parking")

    rows = cursor.fetchall()

    conn.close()

    return rows


def delete_all_data():
    conn = _connect()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM parking")

    conn.commit()
    conn.close()