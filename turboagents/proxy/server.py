"""Proxy server helpers."""

from __future__ import annotations

from typing import Any

from turboagents.proxy.dashboard import DashboardState


def build_app() -> Any:
    try:
        from fastapi import FastAPI
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise RuntimeError(
            "FastAPI is not installed. Install with `pip install turboagents[proxy]`."
        ) from exc

    state = DashboardState(metrics={"status": "starting"})
    app = FastAPI(title="turboagents proxy", version="0.1.0a0")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/dashboard")
    async def dashboard() -> dict[str, Any]:
        return state.snapshot()

    @app.get("/v1/models")
    async def list_models() -> dict[str, list[dict[str, str]]]:
        return {"data": [{"id": "turboagents-placeholder", "object": "model"}]}

    return app


def run_server(host: str = "127.0.0.1", port: int = 8000) -> str:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on optional extras
        raise RuntimeError(
            "uvicorn is not installed. Install with `pip install turboagents[proxy]`."
        ) from exc
    app = build_app()
    uvicorn.run(app, host=host, port=port)
    return "server stopped"
