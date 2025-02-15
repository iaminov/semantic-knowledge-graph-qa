"""
Tests for the graph_builder module.
"""

import pytest
import networkx as nx
from kgqa_agent.graph_builder import build_graph, get_graph_stats, KnowledgeGraphBuilder


class TestKnowledgeGraphBuilder:
    def setup_method(self):
        self.builder = KnowledgeGraphBuilder()

    def test_extract_entities_simple(self):
        text = "John works at Microsoft. He lives in Seattle."
        entities = self.builder.extract_entities(text)
        assert "John" in entities
        assert "Microsoft" in entities
        assert "Seattle" in entities

    def test_extract_entities_quoted(self):
        text = 'The book "Machine Learning Basics" was written by Jane Smith.'
        entities = self.builder.extract_entities(text)
        assert "Machine Learning Basics" in entities
        assert "Jane Smith" in entities

    def test_extract_relations_simple(self):
        text = "John is a developer. Mary has experience."
        entities = {"John", "developer", "Mary", "experience"}
        relations = self.builder.extract_relations(text, entities)
        assert len(relations) >= 1
        found_relation = False
        for subject, predicate, obj in relations:
            if predicate == "is_a" and "john" in subject.lower() and "developer" in obj.lower():
                found_relation = True
        assert found_relation


class TestBuildGraphFunction:
    def test_build_graph_empty_input(self):
        graph = build_graph([])
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() == 0

    def test_build_graph_single_text(self):
        texts = ["Alice is a researcher. She works at Stanford University."]
        graph = build_graph(texts)
        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() > 0


class TestGetGraphStats:
    def test_get_stats_empty_graph(self):
        graph = nx.DiGraph()
        stats = get_graph_stats(graph)
        assert stats['nodes'] == 0
        assert stats['edges'] == 0

    def test_get_stats_simple_graph(self):
        graph = nx.DiGraph()
        graph.add_node("Alice")
        graph.add_node("Bob")
        graph.add_edge("Alice", "Bob", relation="knows")
        stats = get_graph_stats(graph)
        assert stats['nodes'] == 2
        assert stats['edges'] == 1
