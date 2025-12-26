import streamlit as st
import numpy as np
import cv2
from PIL import Image
from insightface.app import FaceAnalysis
from firebase_utils import db
from face_matcher import find_best_match

st.set_page_config("Face Identification", layout="centered")
st.title("🧑‍🏫 Classroom Face Identification")

@st.cache_resource
def load_model():
    model = FaceAnalysis(name="buffalo_l")
    model.prepare(ctx_id=0, det_size=(640, 640))
    return model

model = load_model()

# Load registered users
@st.cache_data
def load_users():
    users = []
    docs = db.collection("users").stream()
    for doc in docs:
        data = doc.to_dict()
        users.append(data)
    return users

registered_users = load_users()

uploaded_file = st.file_uploader("Upload Classroom Image", type=["jpg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    faces = model.get(img_np)

    if len(faces) == 0:
        st.error("No faces detected")
        st.stop()

    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.embedding

        name, score = find_best_match(
            embedding,
            registered_users,
            threshold=0.5
        )

        x1, y1, x2, y2 = bbox
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_np,
            f"{name} ({score:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    st.image(img_np, caption="Identified Faces", use_column_width=True)
