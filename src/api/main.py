"""FastAPI application for SuperDMI Optimization API."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import generator_router, optimization_router, steps_router

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

# Create FastAPI app
app = FastAPI(
    title="Optimizer Dashboard API",
    description="""
    Trading strategy optimization via clustering analysis.

    ## Features

    - **Parameter Landscape Visualization**: Heatmaps showing metric values across parameter combinations
    - **Shortlisting**: Filter variants based on metric thresholds
    - **Feature Engineering**: Automatic scaling and PCA dimensionality reduction
    - **K-Means Clustering**: Group similar strategy variants
    - **HDBSCAN Clustering**: Density-based clustering for refined analysis
    - **Best Cluster Identification**: Automatically identify top-performing clusters

    ## Workflow

    1. Use `/optimization/columns` to get CSV column names for mapping
    2. Use `/optimization/preview` to preview data
    3. Use `/optimization/run` to execute the full pipeline

    All visualizations are returned as Plotly JSON for frontend rendering.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR / "static"), name="static")

# Include routers
app.include_router(optimization_router)
app.include_router(steps_router)
app.include_router(generator_router)


@app.get("/", response_class=HTMLResponse, tags=["frontend"])
async def serve_home():
    """Serve the homepage."""
    html_path = FRONTEND_DIR / "templates" / "home.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/optimizer", response_class=HTMLResponse, tags=["frontend"])
async def serve_optimizer():
    """Serve the optimizer dashboard HTML."""
    html_path = FRONTEND_DIR / "templates" / "optimizer.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/generator", response_class=HTMLResponse, tags=["frontend"])
async def serve_generator():
    """Serve the combination generator HTML."""
    html_path = FRONTEND_DIR / "templates" / "generator.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
