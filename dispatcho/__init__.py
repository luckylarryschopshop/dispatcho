"""Dispatcho Soup — intent classifier for tool operations.

Classifies natural language into tool intents. Fast (<5ms).
No LLM needed for simple commands.

    from dispatcho import classify
    intent, confidence = classify("list the files")
    # ("list_files", 0.99)
"""
from dispatcho.classifier import classify, INTENTS, WORKFLOWS
from dispatcho.param_extractor import extract_params, build_frame_vocab
from dispatcho.router import Router

__version__ = "1.0.0"
