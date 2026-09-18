"""Read-only public projection of deployment-owned builtin Skills."""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from langley.api.dependencies import get_skill_registry
from langley.skill_resources import discover_skill_resources
from langley.skills import (
    SkillConfigurationError,
    SkillIntegrityError,
    SkillRegistry,
    SkillSource,
    read_verified_skill_body,
)

router = APIRouter(prefix="/api/skills", tags=["skills"])


class SkillSummaryResponse(BaseModel):
    name: str
    description: str
    source: SkillSource


class SkillResourceResponse(BaseModel):
    path: str
    byte_size: int


class SkillDetailResponse(SkillSummaryResponse):
    instructions: str
    resources: list[SkillResourceResponse]


def _catalog_unavailable() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": "SKILL_CATALOG_UNAVAILABLE"},
    )


def _content_unavailable() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": "SKILL_CONTENT_UNAVAILABLE"},
    )


@router.get("", response_model=list[SkillSummaryResponse])
def list_builtin_skills(
    registry: SkillRegistry = Depends(get_skill_registry),
) -> list[SkillSummaryResponse]:
    try:
        snapshot = registry.snapshot()
    except (SkillConfigurationError, SkillIntegrityError):
        raise _catalog_unavailable() from None

    return [
        SkillSummaryResponse(
            name=descriptor.name,
            description=descriptor.description,
            source=descriptor.source,
        )
        for descriptor in snapshot.entries
        if descriptor.source is SkillSource.BUILTIN
    ]


@router.get("/{name}", response_model=SkillDetailResponse)
def get_builtin_skill(
    name: str,
    registry: SkillRegistry = Depends(get_skill_registry),
) -> SkillDetailResponse:
    try:
        snapshot = registry.snapshot()
    except (SkillConfigurationError, SkillIntegrityError):
        raise _catalog_unavailable() from None

    descriptor = snapshot.get(name)
    if descriptor is None or descriptor.source is not SkillSource.BUILTIN:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "SKILL_NOT_FOUND"},
        )

    try:
        instructions = read_verified_skill_body(descriptor)
        resources = discover_skill_resources(descriptor.package_root)
    except (SkillConfigurationError, SkillIntegrityError):
        raise _content_unavailable() from None

    return SkillDetailResponse(
        name=descriptor.name,
        description=descriptor.description,
        source=descriptor.source,
        instructions=instructions,
        resources=[
            SkillResourceResponse(
                path=resource.relative_path,
                byte_size=resource.byte_size,
            )
            for resource in resources.entries
        ],
    )
