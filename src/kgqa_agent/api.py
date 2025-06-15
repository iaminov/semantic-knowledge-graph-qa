"""
API Layer

FastAPI application providing REST endpoints for knowledge graph construction and natural language question answering.
"""

import uuid
import networkx as nx
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .graph_builder import build_graph, get_graph_stats
from .query_router import answer_question, get_graph_summary

app = FastAPI(
    title="Semantic Knowledge Graph QA Agent",
    description="A semantic knowledge graph construction and query system using LangChain and LangGraph",
    version="0.1.0"
)

graph_storage: dict[str, dict] = {}


class IngestRequest(BaseModel):
    texts: list[str]
    description: str | None = None


class IngestResponse(BaseModel):
    graph_id: str
    message: str
    stats: dict[str, int]
    created_at: str


class QueryResponse(BaseModel):
    answer: str
    graph_id: str
    question: str
    stats: dict[str, int]


class GraphSummaryResponse(BaseModel):
    graph_id: str
    summary: str
    stats: dict[str, int]


class HealthResponse(BaseModel):
    status: str
    message: str
    active_graphs: int


@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="healthy",
        message="Semantic Knowledge Graph QA Agent is running",
        active_graphs=len(graph_storage)
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_texts(request: IngestRequest):
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        graph_id = str(uuid.uuid4())
        graph = build_graph(request.texts)
        stats = get_graph_stats(graph)
        graph_storage[graph_id] = {
            'graph': graph,
            'texts': request.texts,
            'description': request.description,
            'created_at': datetime.now().isoformat(),
            'stats': stats
        }
        return IngestResponse(
            graph_id=graph_id,
            message=f"Successfully created knowledge graph with {stats['nodes']} entities and {stats['edges']} relationships",
            stats=stats,
            created_at=graph_storage[graph_id]['created_at']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/query/{graph_id}", response_model=QueryResponse)
async def query_graph(
    graph_id: str, 
    question: str = Query(..., description="Natural language question to ask the knowledge graph")
):
    try:
        if graph_id not in graph_storage:
            raise HTTPException(status_code=404, detail=f"Graph with ID {graph_id} not found")
        graph_data = graph_storage[graph_id]
        graph = graph_data['graph']
        answer = answer_question(graph, question)
        return QueryResponse(
            answer=answer,
            graph_id=graph_id,
            question=question,
            stats=graph_data['stats']
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/graphs/{graph_id}/summary", response_model=GraphSummaryResponse)
async def get_graph_summary_endpoint(graph_id: str):
    try:
        if graph_id not in graph_storage:
            raise HTTPException(status_code=404, detail=f"Graph with ID {graph_id} not found")
        graph_data = graph_storage[graph_id]
        graph = graph_data['graph']
        summary = get_graph_summary(graph)
        return GraphSummaryResponse(
            graph_id=graph_id,
            summary=summary,
            stats=graph_data['stats']
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/graphs", response_model=list[dict])
async def list_graphs():
    try:
        graphs_info = []
        for graph_id, graph_data in graph_storage.items():
            graphs_info.append({
                'graph_id': graph_id,
                'description': graph_data.get('description'),
                'created_at': graph_data['created_at'],
                'stats': graph_data['stats'],
                'text_count': len(graph_data['texts'])
            })
        return graphs_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.delete("/graphs/{graph_id}")
async def delete_graph(graph_id: str):
    try:
        if graph_id not in graph_storage:
            raise HTTPException(status_code=404, detail=f"Graph with ID {graph_id} not found")
        del graph_storage[graph_id]
        return {"message": f"Successfully deleted graph {graph_id}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Semantic Knowledge Graph QA Agent",
        "version": "0.1.0",
        "active_graphs": len(graph_storage),
        "total_nodes": sum(data['stats']['nodes'] for data in graph_storage.values()),
        "total_edges": sum(data['stats']['edges'] for data in graph_storage.values())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
