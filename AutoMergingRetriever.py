# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:38:00 2025

@author: DeepakKumar
"""


import chromadb
import openai
import config as conf
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes, get_root_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.storage.docstore import SimpleDocumentStore
from pathlib import Path

# Set up OpenAI API key from config
openai.api_key = conf.openaikey['key']

# Configure global settings for OpenAI embeddings and LLM
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)

# Directories
# PERSIST_DIR = Path("./chroma_db_automerging_openai")
# DATA_DIR = Path("./data/1234567")

# Embedding dimension for text-embedding-3-small
# EXPECTED_DIMENSION = 1536

class Auto_Merging():
    def __init__(self,EXPECTED_DIMENSION,REQUEST_ID):
        self.EXPECTED_DIMENSION = EXPECTED_DIMENSION
        self.PERSIST_DIR = Path("./storage/chroma_db_automerging_openai")
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
        # Validate the data directory
        self.validate_data_directory(self.DATA_DIR)
    
        # Load documents
        documents = SimpleDirectoryReader(input_dir=str(self.DATA_DIR)).load_data()
    
        # Define hierarchical node parser
        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
        automerging_nodes = node_parser.get_nodes_from_documents(documents)
    
        # Print node statistics
        print(f"Total Number of Nodes Parsed: {len(automerging_nodes)}")
        leaf_nodes = get_leaf_nodes(automerging_nodes)
        root_nodes = get_root_nodes(automerging_nodes)
        print(f"Total Number of Leaf Nodes: {len(leaf_nodes)}")
        print(f"Total Number of Root Nodes: {len(root_nodes)}")
    
        # Initialize Chroma persistent client
        chroma_client = chromadb.PersistentClient(path=str(self.PERSIST_DIR))
        collection_name = "automerging_collection_openai"
    
        # Reset collection if it exists to avoid dimensionality issues
        try:
            chroma_client.delete_collection(collection_name)
            print(f"Deleted existing collection {collection_name} to ensure consistency.")
        except:
            pass  # If collection doesn’t exist, proceed
    
        chroma_collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=None,  # LlamaIndex handles embeddings
            metadata={"hnsw:space": "cosine"}
        )
    
        # Set up vector store and storage context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        docstore = SimpleDocumentStore()
        docstore.add_documents(automerging_nodes)  # Add ALL nodes, not just leaf nodes
        storage_context = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)
    
        # Create VectorStoreIndex with leaf nodes (for vector search)
        index = VectorStoreIndex(
            nodes=leaf_nodes,
            storage_context=storage_context,
            show_progress=True,
            store_nodes_override=True  # Ensure all nodes are stored for retrieval
        )
    
        # Persist the index and docstore
        storage_context.persist(persist_dir=str(self.PERSIST_DIR))
        print(f"Docstore contains {len(docstore.docs)} nodes.")
        print(f"Index created and persisted to {self.PERSIST_DIR}")
    
        return index
    
    # Function to load the persisted index and set up the AutoMergingRetriever
    def load_index_and_retriever(self):
        # Initialize Chroma persistent client
        chroma_client = chromadb.PersistentClient(path=str(self.PERSIST_DIR))
        chroma_collection = chroma_client.get_or_create_collection("automerging_collection_openai")
    
        # Set up vector store and storage context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(self.PERSIST_DIR)
        )
    
        # Load the index from storage
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
    
        # Create base retriever
        base_retriever = index.as_retriever(similarity_top_k=6)
    
        # Wrap with AutoMergingRetriever
        automerging_retriever = AutoMergingRetriever(
            base_retriever,
            storage_context=storage_context,
            verbose=True
        )
    
        return automerging_retriever, base_retriever

