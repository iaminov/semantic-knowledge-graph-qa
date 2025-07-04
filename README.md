# Semantic Knowledge Graph QA

An intelligent knowledge graph construction and natural language question answering system using LangChain, LangGraph, and NetworkX.

## Overview

This project builds semantic knowledge graphs from arbitrary text corpora and enables natural language question answering through graph traversal algorithms. The system extracts entities and relationships from text, constructs a directed graph representation, and uses intelligent routing to answer user questions.

## Features

- Text ingestion and processing
- Entity and relationship extraction
- Knowledge graph construction using NetworkX
- Natural language question answering
- REST API with FastAPI
- Docker containerization
- Comprehensive testing suite

## Tech Stack

- Python 3.10+
- LangChain for text processing
- LangGraph for graph operations
- NetworkX for graph representation
- FastAPI for API layer
- Uvicorn for ASGI server
- Pydantic for data validation
- Pytest for testing
- Docker for containerization

## Project Structure

```
semantic-knowledge-graph-qa/
├── src/
│   ├── kgqa_agent/
│   │   ├── __init__.py
│   │   ├── graph_builder.py
│   │   ├── query_router.py
│   │   └── api.py
│   └── tests/
│       ├── test_graph_builder.py
│       └── test_query_router.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/semantic-knowledge-graph-qa.git
cd semantic-knowledge-graph-qa
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Locally

Start the development server:
```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
uvicorn kgqa_agent.api:app --reload --host 0.0.0.0 --port 8000
```

### Using Docker

Build and run with Docker Compose:
```bash
docker-compose up --build
```

### Example Usage

1. Ingest text documents:
```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Google was founded by Larry Page and Sergey Brin in 1998.",
      "Larry Page is a computer scientist and co-founder of Google.",
      "Sergey Brin worked on the PageRank algorithm with Larry Page.",
      "Sundar Pichai is the current CEO of Google."
    ],
    "description": "Information about Google and its founders"
  }'
```

2. Query the knowledge graph:
```bash
curl -X GET "http://localhost:8000/query/YOUR_GRAPH_ID?question=Who%20founded%20Google" \
  -H "accept: application/json"
```

3. Get graph summary:
```bash
curl -X GET "http://localhost:8000/graphs/YOUR_GRAPH_ID/summary" \
  -H "accept: application/json"
```

### API Endpoints

The API will be available at `http://localhost:8000` with the following endpoints:

- `POST /ingest` - Ingest text documents and build knowledge graph
- `GET /query/{graph_id}` - Query knowledge graph with natural language
- `GET /graphs/{graph_id}/summary` - Get graph statistics and summary
- `GET /graphs` - List all available graphs
- `DELETE /graphs/{graph_id}` - Delete a knowledge graph
- `GET /health` - Health check endpoint

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing

Run the test suite:
```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
pytest src/tests/ -v
```

## Development

### Code Quality

The project uses:
- flake8 for linting
- pytest for testing
- GitHub Actions for CI/CD

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## Architecture

### Graph Builder
Handles text processing, entity extraction, and knowledge graph construction. Uses pattern matching and NLP techniques to identify entities and relationships.

### Query Router
Processes natural language questions and routes them through the knowledge graph to find relevant answers. Supports various question types including entity queries and relationship questions.

### API Layer
Provides REST endpoints for graph management and querying. Handles request validation, error handling, and response formatting.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

Built with modern Python tools and libraries including LangChain, LangGraph, and NetworkX.
