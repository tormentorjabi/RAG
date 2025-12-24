from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import os
from typing import List, Optional
import json
from app.vector_store import get_chroma_client, get_collection
from app.embeddings import embed_texts
from app.ingest import load_and_chunk_files
from app.config import CHROMA_COLLECTION_NAME
import chromadb

app = FastAPI()

# Create a directory for uploaded files
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Save the file
        file_location = UPLOAD_DIR / file.filename
        with open(file_location, "wb+") as file_object:
            file_object.write(await file.read())
        
        # Process the document and add to the vector store
        client = get_chroma_client()
        collection = get_collection(client)
        
        # Process the document
        documents = load_and_chunk_files([str(file_location)])
        
        # Add to ChromaDB
        ids = [f"{file.filename}_{i}" for i in range(len(documents))]
        embeddings = embed_texts(documents)
        metadatas = [{"filename": file.filename, "chunk": i} for i in range(len(documents))]
        
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        return {"filename": file.filename, "saved": True, "chunks": len(documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/")
async def search(query: dict):
    try:
        client = get_chroma_client()
        collection = get_collection(client)
        
        # Get query embedding
        query_embedding = embed_texts([query.get("query", "")])[0]
        
        # Search in ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=query.get("top_k", 3)
        )
        
        # Format results
        formatted_results = []
        if results and 'documents' in results:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    "page_content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results.get('metadatas') else {},
                    "distance": results['distances'][0][i] if results.get('distances') else None
                })
        
        return {
            "query": query.get("query"),
            "results": formatted_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
