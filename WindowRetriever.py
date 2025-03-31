
import chromadb
import openai
import config as conf
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceWindowNodeParser
# from llama_index.core.retrievers import SentenceWindowRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from pathlib import Path
from evaluation import RetrieverEvaluator  # Import the updated evaluation module

# Set up OpenAI API key from config
openai.api_key = conf.openaikey['key']

# Configure global settings for OpenAI embeddings and LLM
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)

# Directories
# PERSIST_DIR = Path("./storage/chroma_db_sentence_window_openai")
# DATA_DIR = Path("./data/1234567")

# # Embedding dimension for text-embedding-3-small
# EXPECTED_DIMENSION = 1536

class Window_Retrieval():
    def __init__(self,EXPECTED_DIMENSION,REQUEST_ID):
        self.EXPECTED_DIMENSION = EXPECTED_DIMENSION
        self.PERSIST_DIR = Path("./storage/chroma_db_sentence_window_openai")
        self.REQUEST_ID = REQUEST_ID
        self.DATA_DIR = Path(f"./data/{self.REQUEST_ID}")

    # Function to check if the persisted index is valid
    def is_valid_persisted_index(self,persist_dir):
        required_files = ["docstore.json", "index_store.json", "vector_store.json"]
        return all((persist_dir / file).exists() for file in required_files)
    
    # Function to validate the data directory
    def validate_data_directory(self,data_dir):
        if not data_dir.exists():
            raise ValueError(f"Data directory {data_dir} does not exist. Please create it and add files.")
        if not any(data_dir.iterdir()):
            raise ValueError(f"No files found in {data_dir}. Please add at least one readable file (e.g., .txt, .pdf).")
    
    # Function to create and persist the index
    def create_and_persist_index(self):
        self.validate_data_directory(self.DATA_DIR)
        documents = SimpleDirectoryReader(input_dir=str(self.DATA_DIR)).load_data()
        print(f"Loaded {len(documents)} documents")
    
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        )
        nodes = node_parser.get_nodes_from_documents(documents)
        print(f"Parsed {len(nodes)} nodes")
    
        chroma_client = chromadb.PersistentClient(path=str(self.PERSIST_DIR))
        collection_name = "sentence_window_collection_openai"
        try:
            chroma_client.delete_collection(collection_name)
            print(f"Deleted existing collection {collection_name} to ensure consistency.")
        except:
            pass
    
        chroma_collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"}
        )
    
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=True)
        storage_context.persist(persist_dir=str(self.PERSIST_DIR))
        print(f"Index created and persisted to {self.PERSIST_DIR}")
        return index
    
    # Function to load the persisted index and set up the SentenceWindowRetriever
    def load_index_and_retriever(self):
        chroma_client = chromadb.PersistentClient(path=str(self.PERSIST_DIR))
        chroma_collection = chroma_client.get_or_create_collection("sentence_window_collection_openai")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(self.PERSIST_DIR))
        
        sentence_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
        
        from llama_index.core.postprocessor import MetadataReplacementPostProcessor
        
        sentence_retriever = sentence_index.as_query_engine(
            similarity_top_k=6,
            node_postprocessors=[
                MetadataReplacementPostProcessor(target_metadata_key="window")
            ]
        )
        
        # sentence_retriever = SentenceWindowRetriever(index=index, similarity_top_k=6, window_size=3, verbose=True)
        return sentence_retriever, sentence_index
    
    # Main execution
    # if __name__ == "__main__":
    #     if not PERSIST_DIR.exists() or not is_valid_persisted_index(PERSIST_DIR):
    #         print("Creating new index with OpenAI embeddings...")
    #         index = create_and_persist_index()
    #     else:
    #         print("Valid index already exists, skipping creation.")
    
    #     sentence_retriever, index = load_index_and_retriever()
    #     query_engine = RetrieverQueryEngine.from_args(sentence_retriever)
    
    #     # Test query
    #     query_str = "Find a common grounds for the both documents where each concepts can be utilized. provide references from the documents as well as proof"
    #     print(f"Testing query: {query_str}")
    #     try:
    #         response = query_engine.query(query_str)
    #         print(f"Response: {response}")
    #     except Exception as e:
    #         print(f"Error during query: {e}")
    
    #     # Dynamic evaluation
    #     try:
    #         evaluator = RetrieverEvaluator(data_dir=DATA_DIR)
    #         print("\nEvaluating SentenceWindowRetriever performance...")
    #         evaluation_results = evaluator.evaluate(query_engine=query_engine, num_eval_questions=15)
    #         print("Evaluation completed.")
    #     except Exception as e:
    #         print(f"Error during evaluation: {e}")
    #         print("Evaluation skipped.")
        
        
