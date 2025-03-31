# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 18:39:09 2025

@author: DeepakKumar
"""

# BM25Reranker.py
# BM25Reranker.py
from rank_bm25 import BM25Okapi
from llama_index.core.schema import NodeWithScore, QueryBundle, BaseNode

class BM25Reranker:
    """Class to rerank retrieved nodes using BM25 scoring."""
    
    def __init__(self, top_n=3):
        """Initialize the reranker with a specified top_n value."""
        self.top_n = top_n

    def postprocess_nodes(self, nodes, query_bundle=None):
        """Rerank the retrieved nodes based on BM25 scores using the query from query_bundle."""
        if not nodes or not query_bundle:
            return nodes

        # Extract query string from query_bundle
        query_str = query_bundle.query_str if query_bundle else ""

        # Extract text content from nodes, handling NodeWithScore objects
        corpus = []
        base_nodes = []
        for node in nodes:
            if isinstance(node, NodeWithScore):
                base_node = node.node  # Extract the BaseNode from NodeWithScore
            elif isinstance(node, BaseNode):
                base_node = node  # Already a BaseNode
            else:
                raise ValueError(f"Unexpected node type: {type(node)}")
            corpus.append(base_node.text)
            base_nodes.append(base_node)
        
        if not corpus:
            return nodes  # Fallback if no text available

        # Tokenize the corpus and query (simple whitespace split)
        tokenized_corpus = [doc.split() for doc in corpus]
        tokenized_query = query_str.split()

        # Initialize BM25 model with the tokenized corpus
        bm25 = BM25Okapi(tokenized_corpus)

        # Compute BM25 scores for the query
        scores = bm25.get_scores(tokenized_query)

        # Pair base nodes with their BM25 scores
        node_score_pairs = [
            NodeWithScore(node=base_node, score=score)
            for base_node, score in zip(base_nodes, scores)
        ]

        # Sort by score (descending) and take top_n
        sorted_nodes = sorted(node_score_pairs, key=lambda x: x.score, reverse=True)
        reranked_nodes = sorted_nodes[:self.top_n]

        return reranked_nodes