import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_best_match(face_embedding, registered_users, threshold=0.5):
    """
    face_embedding: np.array (512,)
    registered_users: list of dicts [{name, embedding}]
    """

    best_name = "Unknown"
    best_score = -1

    for user in registered_users:
        stored_emb = np.array(user["embedding"]).reshape(1, -1)
        score = cosine_similarity(
            face_embedding.reshape(1, -1),
            stored_emb
        )[0][0]

        if score > best_score:
            best_score = score
            best_name = user["name"]

    if best_score >= threshold:
        return best_name, float(best_score)
    else:
        return "Unknown", float(best_score)
