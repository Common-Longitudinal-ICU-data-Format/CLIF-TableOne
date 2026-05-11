"""Configuration management routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server import session
from server.config import load_config

router = APIRouter(prefix="/api", tags=["config"])


class ConfigUpdate(BaseModel):
    config_path: str


@router.get("/config")
async def get_config():
    """Return the currently loaded config."""
    config = session.get("config")
    if not config:
        raise HTTPException(404, "No config loaded")
    return config


@router.put("/config")
async def update_config(body: ConfigUpdate):
    """Load a new config from the given path and store it in session."""
    config = load_config(body.config_path)
    session.set("config", config)
    return config
