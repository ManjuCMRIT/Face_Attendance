import numpy as np

def cosine_similarity(emb1, emb2):
    """
    Computes cosine similarity between two embeddings
    """
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)

    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def find_best_match(query_embedding, users_dict, threshold=0.55):
    """
    Finds best match from registered users

    users_dict format:
    {
        "Name1": embedding_list,
        "Name2": embedding_list
    }
    """
    best_match = None
    best_score = 0

    for name, embedding in users_dict.items():
        score = cosine_similarity(query_embedding, embedding)

        if score > best_score:
            best_score = score
            best_match = name

    if best_score >= threshold:
        return best_match, best_score

    return None, best_score
