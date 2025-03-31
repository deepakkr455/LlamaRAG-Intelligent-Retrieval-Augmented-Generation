# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 16:18:42 2025

@author: DeepakKumar
"""

# SubQueryRetrieval.py
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes, get_root_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.storage.docstore import SimpleDocumentStore
from pathlib import Path

class SubQuery_Retrieval:
    """Class to manage a SubQuestionQueryEngine with persistent Chroma DB."""

    def __init__(self, expected_dimension: int, request_id: str):
        self.DATA_DIR = Path(f"./data/{request_id}")
        self.PERSIST_DIR = Path(f"./storage/chroma_db_subquery_openai")
        self.EXPECTED_DIMENSION = expected_dimension
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)

    def is_valid_persisted_index(self):
        required_files = ["docstore.json", "index_store.json", "vector_store.json"]
        return all((self.PERSIST_DIR / file).exists() for file in required_files)

    def validate_data_directory(self):
        if not self.DATA_DIR.exists():
            raise ValueError(f"Data directory {self.DATA_DIR} does not exist.")
        if not any(self.DATA_DIR.iterdir()):
            raise ValueError(f"No files found in {self.DATA_DIR}.")

    def create_and_persist_index(self):
        self.validate_data_directory()
        documents = SimpleDirectoryReader(input_dir=str(self.DATA_DIR)).load_data()

        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
        automerging_nodes = node_parser.get_nodes_from_documents(documents)

        print(f"Total Number of Nodes Parsed: {len(automerging_nodes)}")
        leaf_nodes = get_leaf_nodes(automerging_nodes)
        print(f"Total Number of Leaf Nodes: {len(leaf_nodes)}")

        chroma_client = chromadb.PersistentClient(path=str(self.PERSIST_DIR))
        collection_name = "subquery_collection_openai"
        try:
            chroma_client.delete_collection(collection_name)
            print(f"Deleted existing collection {collection_name}.")
        except:
            pass

        chroma_collection = chroma_client.create_collection(name=collection_name, metadata={"hnsw:space": "l2"})
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        docstore = SimpleDocumentStore()
        docstore.add_documents(automerging_nodes)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)

        index = VectorStoreIndex(
            nodes=leaf_nodes,
            storage_context=storage_context,
            show_progress=True,
            store_nodes_override=True
        )

        storage_context.persist(persist_dir=str(self.PERSIST_DIR))
        print(f"Index created and persisted to {self.PERSIST_DIR}")
        return index

    def load_index_and_retriever(self):
        chroma_client = chromadb.PersistentClient(path=str(self.PERSIST_DIR))
        chroma_collection = chroma_client.get_or_create_collection("subquery_collection_openai")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=str(self.PERSIST_DIR))

        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
        base_retriever = index.as_retriever(similarity_top_k=6)
        automerging_retriever = AutoMergingRetriever(base_retriever, storage_context=storage_context, verbose=True)

        # Create base query engine with a timeout or limit
        base_query_engine = RetrieverQueryEngine.from_args(automerging_retriever)

        # Define tool for subquery engine
        query_engine_tool = QueryEngineTool(
            query_engine=base_query_engine,
            metadata=ToolMetadata(
                name="subquery_tool",
                description="Tool for breaking down complex queries into subquestions."
            )
        )

        # Create subquery engine with a limit on subquestions
        subquery_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=[query_engine_tool],
            llm=Settings.llm,
            verbose=True,
            use_async=False  # Use synchronous execution to avoid threading issues
        )

        return subquery_engine, index

