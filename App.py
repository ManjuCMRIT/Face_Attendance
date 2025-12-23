import streamlit as st
import numpy as np
import cv2
from PIL import Image
from insightface.app import FaceAnalysis
from firebase_utils import db
from datetime import datetime
import json
import firebase_admin
from firebase_admin import credentials

# ---------------- Firebase Init ----------------
if not firebase_admin._apps:
    cred = credentials.Certificate(
        json.loads(st.secrets["FIREBASE_KEY"])
    )
    firebase_admin.initialize_app(cred)

# ---------------- Page Config ----------------
st.set_page_config(page_title="Face Attendance", layout="centered")
st.title("🧠 Face Attendance System")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    model = FaceAnalysis(name="buffalo_l")
    model.prepare(ctx_id=0, det_size=(640, 640))
    return model

model = load_model()

# ---------------- Utils ----------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------- UI ----------------
st.info("📸 Capture face to mark attendance")

camera_image = st.camera_input("Take Attendance Photo")

if camera_image:
    img = Image.open(camera_image).convert("RGB")
    img_np = np.array(img)

    faces = model.get(img_np)

    if len(faces) != 1:
        st.error("❌ Exactly ONE face required")
        st.stop()

    query_embedding = faces[0].embedding

    # Fetch registered users
    users = db.collection("users").stream()

    best_match = None
    best_score = 0

    for user in users:
        data = user.to_dict()
        db_embedding = np.array(data["embedding"])

        score = cosine_similarity(query_embedding, db_embedding)

        if score > best_score:
            best_score = score
            best_match = data["name"]

    # ---------------- Threshold ----------------
    THRESHOLD = 0.55

    if best_score >= THRESHOLD:
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%I:%M %p")

        db.collection("attendance").document(date).set({
            best_match: {
                "time": time,
                "confidence": float(best_score)
            }
        }, merge=True)

        st.success(f"✅ Attendance marked for **{best_match}**")
        st.write(f"Confidence: `{best_score:.2f}`")

    else:
        st.error("❌ Face not recognized")
        st.write(f"Best confidence: `{best_score:.2f}`")
