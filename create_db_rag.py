import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os
import shutil


load_dotenv(override=True)

### Variables ###

DATA_PATH = "Data_sources"
CHROMA_PATH = "chroma_db"


# -----------------------------
# 1)  EMBEDDINGS en utilisant OpenRouter et MiniLM de SentenceTransformers (Meilleur rapport qualité / vitesse / coût)
# -----------------------------

client_embed = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["EMBEDDINGS_API_KEY"],
)


def embed_text(text: str):
    """
    Create an embedding using OpenRouter with MiniLM.
    Returns a Python list of floats.
    """
    response = client_embed.embeddings.create(
        model="sentence-transformers/all-minilm-l12-v2",
        input=text,
        encoding_format="float",
    )

    return response.data[0].embedding


# Wrapper pour Chroma (LangChain prend une classe Embeddings)

class OpenRouterEmbeddings:
    """Transformer embedding en format classe pour Chroma."""

    def embed_documents(self, texts):
        return [embed_text(t) for t in texts]
    # Pour embedder la requête de l'utilisateur
    def embed_query(self, text):
        return embed_text(text)





# -----------------------------
# 2) LOAD PDF DOCUMENTS not pdf pour l'instant trop de travail avec unstructured
# -----------------------------


def load_documents():
    """On load les docs .txt  .md .pdf du dossier DATA_PATH."""
    # Pattern de nos données
    loader_txt = DirectoryLoader(DATA_PATH, glob="*.txt")
    # md_loader = DirectoryLoader(DATA_PATH, glob="*.md")  # trop compliqué pour l'instant
    loader_pdf = DirectoryLoader(DATA_PATH, glob="*.pdf",
                                 loader_cls=PyPDFLoader,
                                loader_kwargs={"extract_images": False} )
    
    # loader les deux types de doc
    documents = loader_txt.load() + loader_pdf.load()
    
    print(f"Loaded {len(documents)} text/markdown/pdf files.")
    return documents


# -----------------------------
# 3) On utilise un text splitter de langchain pour découper les documents en chunks 
# -----------------------------

def split_text(documents: list[Document]):
    """Splitter les documents en chunks plus petits."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


# -----------------------------
# 4) Enregistrer les EMBEDDINGS dans CHROMA
# -----------------------------

embeddings = OpenRouterEmbeddings()

def save_to_chroma(chunks: list[Document]):
    """Delete old database and store new chunks with embeddings."""

    # effacer la base de données existante
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

    return db


# -----------------------------
# 5) Notre PIPELINE
# -----------------------------
def create_vector_db():
    """Analyser PDFs + TXT → diviser en chunks → embed → enregistrer dans Chroma_db."""
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


if __name__ == "__main__":
    docs = load_documents()
    print(docs[0].page_content[:500])  
    chunks = split_text(docs)
    print(chunks[0].page_content)
    save_to_chroma(chunks)
