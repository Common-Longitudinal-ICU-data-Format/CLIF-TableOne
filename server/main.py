"""FastAPI application for CLIF 2.1 Validation."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from server.config import load_config
from server import session


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config on startup."""
    try:
        config = load_config("config/config.json")
        session.set("config", config)
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
    yield


app = FastAPI(title="CLIF 2.1 Validation", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers (with try/except for progressive development)
_route_modules = [
    "server.routes.config_routes",
    "server.routes.tables_routes",
    "server.routes.analysis_routes",
    "server.routes.validation_routes",
    "server.routes.summary_routes",
    "server.routes.feedback_routes",
    "server.routes.reports_routes",
    "server.routes.tableone_routes",
    "server.routes.mcide_routes",
    "server.routes.sse_routes",
]

for module_path in _route_modules:
    try:
        import importlib
        mod = importlib.import_module(module_path)
        if hasattr(mod, "router"):
            app.include_router(mod.router)
    except ImportError:
        pass


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


# Mount images directory
images_dir = Path(__file__).parent.parent / "images"
if images_dir.exists():
    app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")

# Mount static files LAST (catch-all)
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
