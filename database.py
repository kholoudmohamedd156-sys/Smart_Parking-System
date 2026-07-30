"""
database.py
SQLite data layer for the Smart Parking System.
Handles vehicle check-in / check-out, available-spot tracking, and history logs.
"""

import sqlite3
from datetime import datetime
from contextlib import contextmanager

DB_PATH = "parking.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist yet, and seed default settings."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_number TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                status TEXT NOT NULL DEFAULT 'IN',
                image_path TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        cur.execute("SELECT value FROM settings WHERE key = 'total_spots'")
        if cur.fetchone() is None:
            cur.execute(
                "INSERT INTO settings (key, value) VALUES ('total_spots', ?)",
                ("20",),
            )


# ---------- Settings ----------

def get_total_spots() -> int:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = 'total_spots'"
        ).fetchone()
        return int(row["value"]) if row else 20


def set_total_spots(n: int):
    with get_conn() as conn:
        conn.execute(
            "UPDATE settings SET value = ? WHERE key = 'total_spots'", (str(n),)
        )


# ---------- Vehicle operations ----------

def is_currently_parked(plate_number: str) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM vehicles WHERE plate_number = ? AND status = 'IN'",
            (plate_number,),
        ).fetchone()
        return row is not None


def check_in(plate_number: str, image_path: str = None) -> bool:
    """Register a vehicle entry. Returns False if plate is already parked
    or if the lot is full."""
    if is_currently_parked(plate_number):
        return False
    if get_available_spots() <= 0:
        return False

    with get_conn() as conn:
        conn.execute(
            """INSERT INTO vehicles (plate_number, entry_time, status, image_path)
               VALUES (?, ?, 'IN', ?)""",
            (plate_number, datetime.now().isoformat(timespec="seconds"), image_path),
        )
    return True


def check_out(plate_number: str) -> bool:
    """Register a vehicle exit. Returns False if the plate isn't currently parked."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id FROM vehicles WHERE plate_number = ? AND status = 'IN' "
            "ORDER BY entry_time DESC LIMIT 1",
            (plate_number,),
        ).fetchone()
        if row is None:
            return False
        conn.execute(
            "UPDATE vehicles SET status = 'OUT', exit_time = ? WHERE id = ?",
            (datetime.now().isoformat(timespec="seconds"), row["id"]),
        )
    return True


def get_active_vehicles():
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM vehicles WHERE status = 'IN' ORDER BY entry_time DESC"
        ).fetchall()


def get_all_logs(limit: int = 200):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM vehicles ORDER BY entry_time DESC LIMIT ?", (limit,)
        ).fetchall()


def get_available_spots() -> int:
    total = get_total_spots()
    with get_conn() as conn:
        occupied = conn.execute(
            "SELECT COUNT(*) AS c FROM vehicles WHERE status = 'IN'"
        ).fetchone()["c"]
    return max(total - occupied, 0)