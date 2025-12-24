from app.vector_store import get_chroma_client, get_collection
from app.embeddings import embed_texts
from app.logger import logger


def search_query(query: str, top_k: int = 3):
    """
    Поиск запроса в ChromaDB с логированием чанков и метрики.
    """
    logger.info(f"QUERY: {query}")

    client = get_chroma_client()
    collection = get_collection(client)

    # Получаем embedding запроса
    query_embedding = embed_texts([query])[0]

    # Поиск top_k ближайших чанков
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=['documents', 'distances']
    )

    docs = results['documents'][0]
    distances = results['distances'][0]

    # Логируем retrieved chunks
    logger.info("CONTEXT RETRIEVED FROM VECTOR DB:")
    for i, (doc, dist) in enumerate(zip(docs, distances)):
        logger.info(f"CTX-{i+1} | distance={dist:.4f}")
        logger.info(doc)

    # Метрика: среднее расстояние
    mean_distance = sum(distances) / len(distances) if distances else 0
    logger.info(f"Mean distance: {mean_distance:.4f}")

    answer = " ".join(docs)[:500]  # первые 500 символов
    logger.info("FINAL ANSWER:")
    logger.info(answer)

    return docs, answer


if __name__ == "__main__":
    # Пример запроса
    query_text = "Что такое RAG?"
    search_query(query_text)
