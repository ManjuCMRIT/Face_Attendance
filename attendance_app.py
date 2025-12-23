import streamlit as st
import numpy as np
import cv2
from PIL import Image
from insightface.app import FaceAnalysis
from utils.face_quality import is_low_light, is_blurry
from utils.similarity import find_best_match
from firebase_utils import db, get_registered_users
from datetime import datetime

st.set_page_config(page_title="Face Attendance", layout="centered")
st.title("🧠 Face Attendance System")

@st.cache_resource
def load_model():
    model = FaceAnalysis(name="buffalo_l")
    model.prepare(ctx_id=0, det_size=(640, 640))
    return model

model = load_model()

camera_image = st.camera_input("📸 Capture Attendance")

if camera_image:
    img = Image.open(camera_image).convert("RGB")
    img_np = np.array(img)

    faces = model.get(img_np)

    if len(faces) != 1:
        st.error("❌ Exactly ONE face required")
        st.stop()

    face = faces[0]
    bbox = face.bbox

    low_light, brightness = is_low_light(img_np, bbox)
    blurry, blur_score = is_blurry(img_np, bbox)

    if low_light:
        st.error(f"❌ Low light detected (Brightness: {brightness:.1f})")
        st.stop()

    if blurry:
        st.error(f"❌ Blurry image (Sharpness: {blur_score:.1f})")
        st.stop()

    users = get_registered_users()

    name, confidence = find_best_match(face.embedding, users)

    if name:
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%I:%M %p")

        db.collection("attendance").document(date).set({
            name: {
                "time": time,
                "confidence": float(confidence)
            }
        }, merge=True)

        st.success(f"✅ Attendance marked for **{name}**")
        st.write(f"Confidence: `{confidence:.2f}`")

    else:
        st.error("❌ Face not recognized")
