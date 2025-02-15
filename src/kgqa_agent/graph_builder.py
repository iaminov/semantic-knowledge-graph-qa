"""
Graph Builder Module

Builds semantic knowledge graphs from text corpora using LangChain for entity and relation extraction, and NetworkX for graph representation.
"""

import networkx as nx
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter


class KnowledgeGraphBuilder:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def extract_entities(self, text: str) -> set[str]:
        entities = set()
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.update(capitalized_words)
        quoted_phrases = re.findall(r'"([^"]*)"', text)
        entities.update(quoted_phrases)

        stop_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An'}
        entities = {entity for entity in entities if entity not in stop_words and len(entity) > 2}
        return entities

    def extract_relations(self, text: str, entities: set[str]) -> list[tuple[str, str, str]]:
        relations = []
        relation_patterns = [
            (r'(\w+(?:\s+\w+)*)\s+is\s+(?:a|an)\s+(\w+(?:\s+\w+)*)', 'is_a'),
            (r'(\w+(?:\s+\w+)*)\s+has\s+(\w+(?:\s+\w+)*)', 'has'),
            (r'(\w+(?:\s+\w+)*)\s+works\s+at\s+(\w+(?:\s+\w+)*)', 'works_at'),
            (r'(\w+(?:\s+\w+)*)\s+lives\s+in\s+(\w+(?:\s+\w+)*)', 'lives_in'),
            (r'(\w+(?:\s+\w+)*)\s+founded\s+(\w+(?:\s+\w+)*)', 'founded'),
            (r'(\w+(?:\s+\w+)*)\s+created\s+(\w+(?:\s+\w+)*)', 'created'),
        ]
        for pattern, relation_type in relation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                subject = match.group(1).strip()
                obj = match.group(2).strip()
                if any(subject.lower() in entity.lower() for entity in entities) and \
                   any(obj.lower() in entity.lower() for entity in entities):
                    relations.append((subject, relation_type, obj))
        return relations

    def build_graph_from_chunks(self, text_chunks: list[str]) -> nx.DiGraph:
        graph = nx.DiGraph()
        for chunk in text_chunks:
            entities = self.extract_entities(chunk)
            for entity in entities:
                if not graph.has_node(entity):
                    graph.add_node(entity, type='entity')
            relations = self.extract_relations(chunk, entities)
            for subject, predicate, obj in relations:
                subject_match = self._find_best_entity_match(subject, entities)
                obj_match = self._find_best_entity_match(obj, entities)
                if subject_match and obj_match:
                    graph.add_edge(subject_match, obj_match, relation=predicate)
        return graph

    def _find_best_entity_match(self, entity: str, entity_set: set[str]) -> str:
        entity_lower = entity.lower()
        for e in entity_set:
            if e.lower() == entity_lower:
                return e
        for e in entity_set:
            if entity_lower in e.lower() or e.lower() in entity_lower:
                return e
        return None


def build_graph(texts: list[str]) -> nx.DiGraph:
    if not texts:
        return nx.DiGraph()
    builder = KnowledgeGraphBuilder()
    combined_text = "\n\n".join(texts)
    chunks = builder.text_splitter.split_text(combined_text)
    graph = builder.build_graph_from_chunks(chunks)
    return graph


def get_graph_stats(graph: nx.DiGraph) -> dict[str, int]:
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "connected_components": nx.number_weakly_connected_components(graph),
        "density": round(nx.density(graph), 4)
    }
