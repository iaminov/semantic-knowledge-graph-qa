"""
Query Router Module

Processes natural language questions for knowledge graph traversal using LangGraph heuristics.
"""

import networkx as nx
import re
from difflib import SequenceMatcher


class QueryRouter:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.question_patterns = {
            'relationship': [r'what is the relationship between (.*?) and (.*?)\?',
                           r'how is (.*?) related to (.*?)\?'],
            'what_is': [r'what is (.*?)\?', r'what are (.*?)\?'],
            'who_is': [r'who is (.*?)\?', r'who are (.*?)\?']
        }

    def extract_entities_from_question(self, question: str) -> list[str]:
        entities = []
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', question)
        entities.extend(capitalized_words)
        quoted_phrases = re.findall(r'"([^"]*)"', question)
        entities.extend(quoted_phrases)
        question_words = {'What', 'Who', 'Where', 'When', 'How', 'Why', 'Which'}
        entities = [e for e in entities if e not in question_words]
        return entities

    def find_similar_entities(self, entity: str, threshold: float = 0.6) -> list[str]:
        similar_entities = []
        entity_lower = entity.lower()
        for node in self.graph.nodes():
            node_lower = node.lower()
            if entity_lower == node_lower:
                return [node]
            if entity_lower in node_lower or node_lower in entity_lower:
                similar_entities.append(node)
            similarity = SequenceMatcher(None, entity_lower, node_lower).ratio()
            if similarity >= threshold:
                similar_entities.append(node)
        return similar_entities

    def get_entity_information(self, entity: str) -> dict[str, any]:
        if entity not in self.graph.nodes():
            return {}
        info = {
            'entity': entity,
            'attributes': dict(self.graph.nodes[entity]),
            'outgoing_relations': [],
            'incoming_relations': [],
            'neighbors': list(self.graph.neighbors(entity))
        }
        for neighbor in self.graph.successors(entity):
            edge_data = self.graph.edges[entity, neighbor]
            info['outgoing_relations'].append({
                'target': neighbor,
                'relation': edge_data.get('relation', 'unknown')
            })
        for predecessor in self.graph.predecessors(entity):
            edge_data = self.graph.edges[predecessor, entity]
            info['incoming_relations'].append({
                'source': predecessor,
                'relation': edge_data.get('relation', 'unknown')
            })
        return info

    def find_path_between_entities(self, entity1: str, entity2: str) -> list[tuple[str, str, str]]:
        try:
            path_nodes = nx.shortest_path(self.graph, entity1, entity2)
            path_relations = []
            for i in range(len(path_nodes) - 1):
                source = path_nodes[i]
                target = path_nodes[i + 1]
                edge_data = self.graph.edges[source, target]
                relation = edge_data.get('relation', 'connected_to')
                path_relations.append((source, relation, target))
            return path_relations
        except nx.NetworkXNoPath:
            return []

    def classify_question(self, question: str) -> tuple[str, list[str]]:
        question_lower = question.lower()
        for q_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, question_lower)
                if match:
                    if q_type == 'relationship':
                        entities = [match.group(1).strip(), match.group(2).strip()]
                    else:
                        entities = [match.group(1).strip()]
                    return q_type, entities
        entities = self.extract_entities_from_question(question)
        return 'general', entities

    def answer_what_is_question(self, entity: str) -> str:
        similar_entities = self.find_similar_entities(entity)
        if not similar_entities:
            return f"I don't have information about '{entity}' in the knowledge graph."
        target_entity = similar_entities[0]
        info = self.get_entity_information(target_entity)
        if not info:
            return f"I found '{target_entity}' but don't have detailed information about it."
        answer_parts = [f"Based on the knowledge graph, here's what I know about {target_entity}:"]
        if info['outgoing_relations']:
            answer_parts.append("\nRelationships:")
            for rel in info['outgoing_relations']:
                answer_parts.append(f"- {target_entity} {rel['relation']} {rel['target']}")
        if info['incoming_relations']:
            answer_parts.append("\nIs related to:")
            for rel in info['incoming_relations']:
                answer_parts.append(f"- {rel['source']} {rel['relation']} {target_entity}")
        if not info['outgoing_relations'] and not info['incoming_relations']:
            answer_parts.append("\nNo specific relationships found in the knowledge graph.")
        return "".join(answer_parts)

    def answer_relationship_question(self, entity1: str, entity2: str) -> str:
        similar_entities1 = self.find_similar_entities(entity1)
        similar_entities2 = self.find_similar_entities(entity2)
        if not similar_entities1:
            return f"I don't have information about '{entity1}' in the knowledge graph."
        if not similar_entities2:
            return f"I don't have information about '{entity2}' in the knowledge graph."
        target_entity1 = similar_entities1[0]
        target_entity2 = similar_entities2[0]
        if self.graph.has_edge(target_entity1, target_entity2):
            edge_data = self.graph.edges[target_entity1, target_entity2]
            relation = edge_data.get('relation', 'connected_to')
            return f"{target_entity1} {relation} {target_entity2}."
        if self.graph.has_edge(target_entity2, target_entity1):
            edge_data = self.graph.edges[target_entity2, target_entity1]
            relation = edge_data.get('relation', 'connected_to')
            return f"{target_entity2} {relation} {target_entity1}."
        path = self.find_path_between_entities(target_entity1, target_entity2)
        if path:
            path_description = " -> ".join([f"{source} ({relation}) {target}" for source, relation, target in path])
            return f"There is an indirect relationship: {path_description}"
        return f"No relationship found between {target_entity1} and {target_entity2} in the knowledge graph."

    def answer_general_question(self, entities: list[str]) -> str:
        if not entities:
            return "I need more specific information to answer your question."
        relevant_info = []
        for entity in entities:
            similar_entities = self.find_similar_entities(entity)
            if similar_entities:
                target_entity = similar_entities[0]
                info = self.get_entity_information(target_entity)
                if info:
                    relevant_info.append(f"**{target_entity}**: Connected to {', '.join(info['neighbors'][:5])}")
        if not relevant_info:
            return "I couldn't find relevant information for the entities mentioned in your question."
        return "Here's what I found:\n" + "\n".join(relevant_info)


def answer_question(graph: nx.DiGraph, question: str) -> str:
    if not graph or graph.number_of_nodes() == 0:
        return "The knowledge graph is empty. Please ingest some text first."
    router = QueryRouter(graph)
    question_type, entities = router.classify_question(question)
    if question_type == 'what_is' and entities:
        return router.answer_what_is_question(entities[0])
    elif question_type == 'who_is' and entities:
        return router.answer_what_is_question(entities[0])
    elif question_type == 'relationship' and len(entities) >= 2:
        return router.answer_relationship_question(entities[0], entities[1])
    else:
        return router.answer_general_question(entities)


def get_graph_summary(graph: nx.DiGraph) -> str:
    if not graph or graph.number_of_nodes() == 0:
        return "The knowledge graph is empty."
    summary_parts = [
        f"Knowledge Graph Summary:",
        f"- {graph.number_of_nodes()} entities",
        f"- {graph.number_of_edges()} relationships",
        f"- {nx.number_weakly_connected_components(graph)} connected components"
    ]
    if graph.number_of_nodes() > 0:
        node_degrees = dict(graph.degree())
        top_entities = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        summary_parts.append("\nTop entities by connections:")
        for entity, degree in top_entities:
            summary_parts.append(f"- {entity} ({degree} connections)")
    return "\n".join(summary_parts)
