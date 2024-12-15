"""
Intelligent Knowledge Graph QA Agent

A semantic knowledge graph construction and query system using LangChain and LangGraph.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .graph_builder import build_graph
from .query_router import answer_question

__all__ = ["build_graph", "answer_question"]
