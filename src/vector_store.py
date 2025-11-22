import chromadb
from chromadb.utils import embedding_functions
from src.fs_utils import get_root_dir

vector_dir_path = get_root_dir() / "data" / "vector_store_persistent_dir"
client = chromadb.PersistentClient(vector_dir_path)

embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def get_collection(name: str):
    return client.get_or_create_collection(name, embedding_function=embed_fn) # type: ignore
