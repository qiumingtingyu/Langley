"""Health endpoint."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def get_health() -> dict[str, str]:
    """Report whether the FastAPI application can serve HTTP requests."""

    return {"status": "ok", "service": "langley"}
