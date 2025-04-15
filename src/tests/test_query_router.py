"""
Tests for the query_router module.
"""

import pytest
import networkx as nx
from kgqa_agent.query_router import QueryRouter, answer_question


@pytest.fixture
def sample_graph():
    graph = nx.DiGraph()
    graph.add_node("Google", type="Company")
    graph.add_node("Larry Page", type="Person")
    graph.add_node("Sergey Brin", type="Person")
    graph.add_node("Sundar Pichai", type="Person")
    graph.add_edge("Larry Page", "Google", relation="co-founded")
    graph.add_edge("Sergey Brin", "Google", relation="co-founded")
    graph.add_edge("Sundar Pichai", "Google", relation="is CEO of")
    return graph


class TestQueryRouter:
    def test_answer_what_is_question(self, sample_graph):
        router = QueryRouter(sample_graph)
        answer = router.answer_what_is_question("Google")
        assert "Google" in answer
        assert "co-founded" in answer.lower()

    def test_answer_relationship_question(self, sample_graph):
        router = QueryRouter(sample_graph)
        answer = router.answer_relationship_question("Larry Page", "Google")
        assert "co-founded" in answer.lower()

    def test_answer_nonexistent_entity(self, sample_graph):
        router = QueryRouter(sample_graph)
        answer = router.answer_what_is_question("Unknown Entity")
        assert "don't have information" in answer.lower()

    def test_classify_question(self, sample_graph):
        router = QueryRouter(sample_graph)
        q_type, entities = router.classify_question("What is the relationship between Google and Larry Page?")
        assert q_type == "relationship"
        assert len(entities) == 2


def test_answer_question_with_empty_graph():
    graph = nx.DiGraph()
    answer = answer_question(graph, "What is Google?")
    assert "empty" in answer.lower()
