import os
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
import chromadb
from chromadb.config import Settings
from app.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR

def get_chroma_client():
    return chromadb.HttpClient(
        host="chroma-db",  # Имя сервиса из docker-compose.yml
        port=8000,          # Порт по умолчанию для ChromaDB
        ssl=False           # Отключаем SSL для локальной разработки
    )

def get_collection(client):
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME
    )
