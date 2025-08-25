import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import os
import dotenv

dotenv.load_dotenv()

DATAPATH = "Data"
CHROMA_PATH = "chroma"

def load_pdf():
    document_loader = PyPDFDirectoryLoader(DATAPATH)
    return document_loader.load()

def get_embedding():
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",  # or another supported model
        openai_api_key=os.getenv("OPENAI_API_KEY")  # or set via environment variable
    )
    return embedding_model

def split_text(documents: list[Document]):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    document = chunks[10]
    # print(document.page_content)

    return chunks

def save_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    database = Chroma.from_documents(
        chunks, embedding=get_embedding(), persist_directory=CHROMA_PATH
    )
    database.persist()
    # print(f"saved {len(chunks)} to {CHROMA_PATH}")

def data_store():
    documents = load_pdf()
    chunks = split_text(documents)
    save_chroma(chunks)


def search_chroma(query):
    embedding_model = get_embedding()

    # data_store()
    database = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_model)
    results = database.similarity_search_with_relevance_scores(query,k=3)

    
    if results[0][1] < 0.2:
        print("no results found")
        return
    
    sourced_chunks = {}

    for result in results:
        doc = result[0]
        score = result[1]
        source = doc.metadata.get("source", "unknown")

        sourced_chunks[score] = {
            "document": doc,
            "source": source
        }


      
    return sourced_chunks


search_chroma("build a hotel")

