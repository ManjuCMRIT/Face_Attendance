import firebase_admin
from firebase_admin import credentials, firestore
import json
import streamlit as st

if not firebase_admin._apps:
    cred = credentials.Certificate(
        json.loads(st.secrets["FIREBASE_KEY"])
    )
    firebase_admin.initialize_app(cred)

db = firestore.client()


def get_registered_users():
    """
    Returns dictionary of users and embeddings
    """
    users_ref = db.collection("users").stream()
    users = {}

    for user in users_ref:
        data = user.to_dict()
        users[data["name"]] = data["embedding"]

    return users
