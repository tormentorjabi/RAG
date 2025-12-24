from pathlib import Path
from app.logger import logger
from app.embeddings import embed_texts
from app.vector_store import get_chroma_client, get_collection

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize


def chunk_text_sentences(text: str, chunk_size: int = 300, overlap: int = 50):
    """
    Разбиваем текст на чанки по предложениям с указанным overlap.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sent in sentences:
        if len(current_chunk) + len(sent) + 1 > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = current_chunk[-overlap:] + " " + sent  # overlap
        else:
            current_chunk += " " + sent

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def load_and_chunk_files(file_paths: list[str], chunk_size: int = 300, overlap: int = 50, split_sentences: bool = True):
    """
    Загружаем несколько файлов и создаём чанки.
    split_sentences=True → разбивает на предложения
    split_sentences=False → весь файл как один chunk
    """
    all_chunks = []
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"File not found: {file_path}")
            continue

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            logger.warning(f"File is empty: {file_path}")
            continue

        if split_sentences:
            chunks = chunk_text_sentences(text, chunk_size, overlap)
        else:
            chunks = [text]  # весь файл как один chunk

        logger.info(f"{len(chunks)} chunks created from {file_path}")
        all_chunks.extend(chunks)

    return all_chunks


def ingest_documents(file_paths: list[str], chunk_size: int = 300, overlap: int = 50, split_sentences: bool = True):
    """
    Ингестим несколько документов в ChromaDB с embeddings.
    """
    chunks = load_and_chunk_files(file_paths, chunk_size, overlap, split_sentences)
    if not chunks:
        logger.warning("No chunks to ingest. Exiting.")
        return

    logger.info(f"Total chunks to ingest: {len(chunks)}")

    embeddings = embed_texts(chunks)
    client = get_chroma_client()

    # Удаляем коллекцию полностью перед ingest
    try:
        client.delete_collection(name="documents")
        logger.info("Collection 'documents' deleted")
    except Exception:
        pass

    # Создаём коллекцию заново
    collection = get_collection(client)
    logger.info(f"Collection '{collection.name}' created")

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

    for i, chunk in enumerate(chunks):
        logger.info(f"Stored chunk {ids[i]}: {chunk[:100]}...")


if __name__ == "__main__":
    # Пример: 3 файла
    file_list = [
        "data/doc1.txt",
        "data/doc2.txt",
        "data/doc3.txt",
        "data/doc4.txt"
    ]
    ingest_documents(file_list, chunk_size=300, overlap=50, split_sentences=True)
