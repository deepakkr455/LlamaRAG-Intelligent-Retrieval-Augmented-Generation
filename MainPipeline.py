# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 16:32:12 2025

@author: DeepakKumar
"""


import threading
import time
from queue import Queue
from WindowRetriever import Window_Retrieval
from AutoRetrieval import Auto_Retrieval
from AutoMergingRetriever import Auto_Merging
from SubQueryRetrieval import SubQuery_Retrieval
from evaluation import RetrieverEvaluator
from BM25Reranker import BM25Reranker  # Import the new BM25 reranker
from llama_index.core.query_engine import RetrieverQueryEngine
from pathlib import Path
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
NUM_EVAL_QUESTIONS = 15
DEFAULT_REQUEST_ID = "1234567"
EXPECTED_DIMENSION = 1536
APPLY_RERANKER = False  # Toggle to enable/disable BM25 reranker

# Wrapper functions for class-based retrievers
def load_sentence_window(apply_reranker=False):
    wr = Window_Retrieval(EXPECTED_DIMENSION, DEFAULT_REQUEST_ID)
    if not wr.is_valid_persisted_index(wr.PERSIST_DIR):
        logger.info(f"Creating index for SentenceWindow...")
        wr.create_and_persist_index()
    query_engine, index = wr.load_index_and_retriever()
    if apply_reranker:
        reranker = BM25Reranker(top_n=3)
        query_engine = RetrieverQueryEngine.from_args(query_engine.retriever, node_postprocessors=[reranker])
    return query_engine, index

def load_auto_retrieval(apply_reranker=False):
    try:
        ar = Auto_Retrieval(EXPECTED_DIMENSION, DEFAULT_REQUEST_ID)
        if not ar.is_valid_persisted_index(ar.PERSIST_DIR):
            logger.info(f"Creating index for AutoRetrieval...")
            ar.create_and_persist_index()
        retriever, index = ar.load_index_and_retriever()
        if apply_reranker:
            reranker = BM25Reranker(top_n=3)
            return RetrieverQueryEngine.from_args(retriever, node_postprocessors=[reranker]), index
        return retriever, index
    except Exception as e:
        logger.error(f"Error in load_auto_retrieval: {e}")
        raise

def load_auto_merging(apply_reranker=False):
    try:
        am = Auto_Merging(EXPECTED_DIMENSION, DEFAULT_REQUEST_ID)
        if not am.is_valid_persisted_index(am.PERSIST_DIR):
            logger.info(f"Creating index for AutoMerging...")
            am.create_and_persist_index()
        retriever, index = am.load_index_and_retriever()
        if apply_reranker:
            reranker = BM25Reranker(top_n=3)
            return RetrieverQueryEngine.from_args(retriever, node_postprocessors=[reranker]), index
        return retriever, index
    except Exception as e:
        logger.error(f"Error in load_auto_merging: {e}")
        raise

def load_subquery(apply_reranker=False):
    try:
        sq = SubQuery_Retrieval(EXPECTED_DIMENSION, DEFAULT_REQUEST_ID)
        if not sq.is_valid_persisted_index():
            logger.info(f"Creating index for SubQuery...")
            sq.create_and_persist_index()
        query_engine, index = sq.load_index_and_retriever()
        if apply_reranker:
            reranker = BM25Reranker(top_n=3)
            query_engine = RetrieverQueryEngine.from_args(query_engine.retriever, node_postprocessors=[reranker])
        return query_engine, index
    except Exception as e:
        logger.error(f"Error in load_subquery: {e}")
        raise

RETRIEVER_TYPES = {
    "AutoMerging": (load_auto_merging, Path(f"./data/{DEFAULT_REQUEST_ID}")),
    "AutoRetrieval": (load_auto_retrieval, Path(f"./data/{DEFAULT_REQUEST_ID}")),
    "SentenceWindow": (load_sentence_window, Path(f"./data/{DEFAULT_REQUEST_ID}")),
    "SubQuery": (load_subquery, Path(f"./data/{DEFAULT_REQUEST_ID}"))
}

results_queue = Queue()

def evaluate_retriever(retriever_name, load_func, data_dir, queue, apply_reranker=False, timeout=120):
    """Evaluate a single retriever with optional BM25 reranker."""
    def run_evaluation():
        start_time = time.time()
        try:
            logger.info(f"Starting evaluation for {retriever_name} (Reranker: {apply_reranker})...")
            retriever_or_engine, index = load_func(apply_reranker=apply_reranker)
            query_engine = (retriever_or_engine if retriever_name in ["SentenceWindow", "SubQuery"]
                           else RetrieverQueryEngine.from_args(retriever_or_engine))
            evaluator = RetrieverEvaluator(data_dir=data_dir)
            logger.info(f"Running evaluation for {retriever_name} with {NUM_EVAL_QUESTIONS} questions...")
            eval_results = evaluator.evaluate(query_engine=query_engine, num_eval_questions=NUM_EVAL_QUESTIONS)
            latency = time.time() - start_time
            
            queue.put({
                "Retriever": retriever_name,
                "Faithfulness": eval_results["faithfulness"],
                "Relevancy": eval_results["relevancy"],
                "Correctness": eval_results["correctness"],
                "Latency": latency,
                "AvgLatency": latency / NUM_EVAL_QUESTIONS,
                "Reranker": apply_reranker
            })
            logger.info(f"Completed evaluation for {retriever_name} in {latency:.2f}s")
        except Exception as e:
            queue.put({
                "Retriever": retriever_name,
                "Faithfulness": 0.0,
                "Relevancy": 0.0,
                "Correctness": 0.0,
                "Latency": 0.0,
                "AvgLatency": 0.0,
                "Error": str(e),
                "Reranker": apply_reranker
            })
            logger.error(f"Error evaluating {retriever_name}: {e}")

    thread = threading.Thread(target=run_evaluation)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        queue.put({
            "Retriever": retriever_name,
            "Faithfulness": 0.0,
            "Relevancy": 0.0,
            "Correctness": 0.0,
            "Latency": 0.0,
            "AvgLatency": 0.0,
            "Error": f"Evaluation timed out after {timeout} seconds",
            "Reranker": apply_reranker
        })
        logger.error(f"Evaluation for {retriever_name} timed out after {timeout}s")

def run_parallel_evaluations():
    """Run evaluations with and without BM25 reranker sequentially."""
    results = []
    start_time = time.time()
    
    # Evaluate without reranker
    logger.info("Starting evaluations without reranker...")
    for retriever_name, (load_func, data_dir) in RETRIEVER_TYPES.items():
        evaluate_retriever(retriever_name, load_func, data_dir, results_queue, apply_reranker=False)
        while not results_queue.empty():
            results.append(results_queue.get())

    # Evaluate with reranker if APPLY_RERANKER is True
    if APPLY_RERANKER:
        logger.info("Starting evaluations with BM25 reranker...")
        for retriever_name, (load_func, data_dir) in RETRIEVER_TYPES.items():
            evaluate_retriever(retriever_name, load_func, data_dir, results_queue, apply_reranker=True)
            while not results_queue.empty():
                results.append(results_queue.get())

    total_time = time.time() - start_time
    logger.info(f"Total evaluation time: {total_time:.2f}s")
    return results

def display_results(results):
    """Display evaluation results with and without BM25 reranker."""
    col_widths = {
        "Retriever": 15,
        "Reranker": 10,
        "Faithfulness": 12,
        "Relevancy": 10,
        "Correctness": 12,
        "Latency": 15,
        "AvgLatency": 15
    }
    
    header = (
        f"{'Retriever':<{col_widths['Retriever']}}|"
        f"{'Reranker':<{col_widths['Reranker']}}|"
        f"{'Faithfulness':<{col_widths['Faithfulness']}}|"
        f"{'Relevancy':<{col_widths['Relevancy']}}|"
        f"{'Correctness':<{col_widths['Correctness']}}|"
        f"{'Total Latency (s)':<{col_widths['Latency']}}|"
        f"{'Avg Latency (s)':<{col_widths['AvgLatency']}}"
    )
    separator = "-" * (sum(col_widths.values()) + len(col_widths))
    
    rows = []
    for result in results:
        row = (
            f"{result['Retriever']:<{col_widths['Retriever']}}|"
            f"{'BM25' if result['Reranker'] else 'No':<{col_widths['Reranker']}}|"
            f"{result['Faithfulness']:<{col_widths['Faithfulness']}}|"
            f"{result['Relevancy']:<{col_widths['Relevancy']}}|"
            f"{result['Correctness']:<{col_widths['Correctness']}}|"
            f"{result['Latency']:<{col_widths['Latency']}}|"
            f"{result['AvgLatency']:<{col_widths['AvgLatency']}}"
        )
        rows.append(row)
    
    print("\n=== Retriever Performance Comparison (With vs Without BM25 Reranker) ===")
    print(separator)
    print(header)
    print(separator)
    for row in rows:
        print(row)
    print(separator)

    for result in results:
        if "Error" in result:
            logger.error(f"Error in {result['Retriever']} (Reranker: {result['Reranker']}): {result['Error']}")

class RetrieverSwitcher:
    """Class to switch between retrievers for querying."""
    def __init__(self):
        self.query_engines = {}
        self.load_retrievers()

    def load_retrievers(self):
        """Load all retrievers and create query engines."""
        for retriever_name, (load_func, _) in RETRIEVER_TYPES.items():
            try:
                retriever_or_engine, _ = load_func(apply_reranker=APPLY_RERANKER)
                self.query_engines[retriever_name] = (
                    retriever_or_engine if retriever_name in ["SentenceWindow", "SubQuery"]
                    else RetrieverQueryEngine.from_args(retriever_or_engine)
                )
                logger.info(f"Loaded {retriever_name} retriever")
            except Exception as e:
                logger.error(f"Failed to load {retriever_name}: {e}")

    def query(self, query_str, retriever_name=None):
        """Query using a specific retriever or all retrievers."""
        if retriever_name:
            if retriever_name in self.query_engines:
                try:
                    response = self.query_engines[retriever_name].query(query_str)
                    return {retriever_name: str(response)}
                except Exception as e:
                    return {retriever_name: f"Error: {e}"}
            else:
                return {retriever_name: "Retriever not found"}
        else:
            results = {}
            for name, engine in self.query_engines.items():
                try:
                    response = engine.query(query_str)
                    results[name] = str(response)
                except Exception as e:
                    results[name] = f"Error: {e}"
            return results

def main():
    print("Starting evaluations...")
    results = run_parallel_evaluations()
    display_results(results)

    switcher = RetrieverSwitcher()
    while True:
        print("\n=== Query Interface ===")
        print("Available retrievers:")
        for i, retriever_name in enumerate(RETRIEVER_TYPES.keys(), 1):
            print(f"  {i}. {retriever_name}")
        print("Enter 'exit' to quit")
        query = input("Enter your query: ")
        if query.lower() == "exit":
            break
        
        retriever_choice = input(
            "Enter retriever name (e.g., 'AutoMerging') or number (e.g., '1'), or press Enter for all: "
        ).strip()
        
        if retriever_choice:
            try:
                choice_num = int(retriever_choice)
                if 1 <= choice_num <= len(RETRIEVER_TYPES):
                    retriever_choice = list(RETRIEVER_TYPES.keys())[choice_num - 1]
                else:
                    print(f"Invalid number. Please choose between 1 and {len(RETRIEVER_TYPES)}.")
                    continue
            except ValueError:
                if retriever_choice not in RETRIEVER_TYPES:
                    print(f"Invalid retriever name '{retriever_choice}'. Please choose from the list above.")
                    continue
            result = switcher.query(query, retriever_choice)
        else:
            result = switcher.query(query)
        
        print("\nQuery Results:")
        for retriever, response in result.items():
            print(f"{retriever}:")
            print(f"  {response}")
            print("-" * 50)

if __name__ == "__main__":
    main()
    
    


