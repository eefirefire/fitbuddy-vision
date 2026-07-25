"""
FitBuddy-AI — Account & Pre-Exercise Intake Module
===================================================

Two responsibilities, kept together because the second depends on the first:

1. A minimal username/password account system (SQLite + salted password
   hashes via werkzeug.security — no plaintext passwords are ever stored).
2. A pre-exercise "intake" — age, recent surgery/diagnosis, current pain
   level, swelling, doctor clearance — collected once per session and used
   to drive the warning system in rehab_knee_extension.py / the frontend.

Every numeric threshold below is sourced (see comments at each constant) —
none of these were invented for this project. Where the literature does NOT
give exercise-specific numbers (this happens for the "movement too fast"
check), that is stated explicitly rather than presented as a hard clinical
limit.
"""

import os
import sqlite3
from functools import wraps

from flask import Blueprint, jsonify, request, session
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fitbuddy_users.db")

auth_bp = Blueprint("auth", __name__)


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS intakes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            age INTEGER NOT NULL,
            surgery_recency TEXT NOT NULL,
            diagnosis TEXT NOT NULL,
            pain_level INTEGER NOT NULL,
            swelling INTEGER NOT NULL,
            cleared_by_doctor INTEGER NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def login_required(view_fn):
    @wraps(view_fn)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Not logged in."}), 401
        return view_fn(*args, **kwargs)
    return wrapped


# ── Account routes ────────────────────────────────────────────────────────────

@auth_bp.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    conn = _get_db()
    try:
        password_hash = generate_password_hash(password)
        cur = conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash),
        )
        conn.commit()
        session["user_id"] = cur.lastrowid
        session["username"] = username
        return jsonify({"username": username})
    except sqlite3.IntegrityError:
        return jsonify({"error": "That username is already taken."}), 409
    finally:
        conn.close()


@auth_bp.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    conn = _get_db()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()

    # Deliberately identical error for "no such user" and "wrong password" —
    # distinguishing them lets an attacker enumerate valid usernames.
    if user is None or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Incorrect username or password."}), 401

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    return jsonify({"username": user["username"]})


@auth_bp.route("/api/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


@auth_bp.route("/api/auth/me")
def me():
    if "user_id" not in session:
        return jsonify({"error": "Not logged in."}), 401
    return jsonify({"username": session["username"]})


# ── Intake routes ─────────────────────────────────────────────────────────────

# CDC's STEADI falls-prevention initiative uses 65+ as the standard
# older-adult screening threshold (age-related decline in reaction time,
# proprioception, and coordination accelerates past this point).
# Source: https://www.cdc.gov/steadi/index.html
OLDER_ADULT_AGE_THRESHOLD = 65

# A widely-used physical-therapy "traffic light" pain rule for exercising
# through discomfort. Sources:
# https://rhphysiotherapy.com/pain-rule-for-exercise/
# https://theprehabguys.com/is-no-pain-no-gain-true/
PAIN_GREEN_MAX = 3   # 0-3/10: proceed
PAIN_AMBER_MAX = 6   # 4-6/10: proceed with caution
# 7+/10 is the "red" zone: stop / do not begin.


def _build_intake_warnings(age, surgery_recency, diagnosis, pain_level, swelling, cleared_by_doctor):
    warnings = []

    if not cleared_by_doctor:
        warnings.append({
            "level": "red",
            "message": "You haven't confirmed medical clearance to exercise. Check with your doctor or physical therapist before starting.",
        })

    if pain_level >= 7:
        warnings.append({
            "level": "red",
            "message": f"Reported pain level ({pain_level}/10) is in the 'stop' range of the standard PT pain traffic-light system (7+/10 = stop). Do not begin this session.",
            "source": "https://rhphysiotherapy.com/pain-rule-for-exercise/",
        })
    elif pain_level >= 4:
        warnings.append({
            "level": "amber",
            "message": f"Reported pain level ({pain_level}/10) is in the 'caution' range (4-6/10). Proceed gently and stop if pain increases.",
            "source": "https://rhphysiotherapy.com/pain-rule-for-exercise/",
        })

    if swelling:
        warnings.append({
            "level": "amber",
            "message": "You reported visible swelling. Swelling is commonly used as a sign to scale back intensity or range of motion until it subsides — check with your PT.",
        })

    if age >= OLDER_ADULT_AGE_THRESHOLD:
        warnings.append({
            "level": "amber",
            "message": f"Age {age} is at or above the CDC's standard threshold (65+) used for fall-risk screening (STEADI initiative). Consider having someone nearby and a stable chair.",
            "source": "https://www.cdc.gov/steadi/index.html",
        })

    if surgery_recency in ("0-2 weeks", "2-4 weeks") and diagnosis != "none":
        warnings.append({
            "level": "amber",
            "message": "You're early in post-surgical recovery. Follow your surgeon/physical therapist's specific motion and weight-bearing restrictions — this app does not know your individual protocol and should not override it.",
        })

    return warnings


@auth_bp.route("/api/intake", methods=["POST"])
@login_required
def submit_intake():
    data = request.get_json(silent=True) or {}
    try:
        age = int(data.get("age"))
        pain_level = int(data.get("pain_level"))
    except (TypeError, ValueError):
        return jsonify({"error": "age and pain_level must be numbers."}), 400

    surgery_recency = data.get("surgery_recency") or "none"
    diagnosis = data.get("diagnosis") or "none"
    swelling = bool(data.get("swelling"))
    cleared_by_doctor = bool(data.get("cleared_by_doctor"))

    if not (0 <= age <= 120):
        return jsonify({"error": "age must be between 0 and 120."}), 400
    if not (0 <= pain_level <= 10):
        return jsonify({"error": "pain_level must be between 0 and 10."}), 400

    conn = _get_db()
    conn.execute(
        """INSERT INTO intakes
           (user_id, age, surgery_recency, diagnosis, pain_level, swelling, cleared_by_doctor)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (session["user_id"], age, surgery_recency, diagnosis, pain_level, int(swelling), int(cleared_by_doctor)),
    )
    conn.commit()
    conn.close()

    # Cache on the session so rehab routes can reference it without a DB hit.
    session["intake"] = {
        "age": age, "surgery_recency": surgery_recency, "diagnosis": diagnosis,
        "pain_level": pain_level, "swelling": swelling, "cleared_by_doctor": cleared_by_doctor,
    }

    warnings = _build_intake_warnings(age, surgery_recency, diagnosis, pain_level, swelling, cleared_by_doctor)
    blocking = any(w["level"] == "red" for w in warnings)
    return jsonify({"warnings": warnings, "blocking": blocking})


@auth_bp.route("/api/intake/current")
@login_required
def current_intake():
    intake = session.get("intake")
    if not intake:
        return jsonify({"error": "No intake submitted this session."}), 404
    return jsonify(intake)
