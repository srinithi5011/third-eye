import torch


def cosine_similarity(embedding_a, embedding_b):
    """
    Calculate cosine similarity between two embeddings.
    """

    embedding_a = embedding_a / embedding_a.norm(
        dim=1,
        keepdim=True
    )

    embedding_b = embedding_b / embedding_b.norm(
        dim=1,
        keepdim=True
    )

    similarity = torch.mm(
        embedding_a,
        embedding_b.T
    )

    return similarity


def recognize(embedding, gallery, threshold=0.5):
    """
    Compare an embedding against the gallery.

    Returns:
        identity
        similarity score
    """

    best_identity = "Unknown"
    best_score = -1.0

    for identity, gallery_embedding in gallery.items():

        score = cosine_similarity(
            embedding,
            gallery_embedding
        ).item()

        if score > best_score:
            best_score = score
            best_identity = identity

    if best_score < threshold:
        best_identity = "Unknown"

    return best_identity, best_score