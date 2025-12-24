import logging
from pathlib import Path

# Создаём папку logs, если её нет
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_file = log_dir / "vector_rag.log"

logger = logging.getLogger("vector_rag")
logger.setLevel(logging.INFO)

# Добавляем обработчики только если их нет
if not logger.hasHandlers():
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Консоль
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Файл
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
