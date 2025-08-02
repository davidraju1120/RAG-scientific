import time
from typing import List, Dict

def evaluate_retrieval_accuracy(retrieved: List[str], relevant: List[str]) -> float:
    if not relevant:
        return 0.0
    correct = sum([1 for doc in retrieved if doc in relevant])
    return correct / len(relevant)

def measure_latency(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        latency = time.time() - start
        return result, latency
    return wrapper
