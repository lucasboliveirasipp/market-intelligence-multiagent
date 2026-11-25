"""Shared state definitions for the LangGraph supervisor workflow."""

from __future__ import annotations

from operator import add
from typing import Annotated, List, Optional

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# Backward compatibility with earlier indexing pipeline ---------------------


class IndexRefs(BaseModel):
    page_start: int = Field(..., description="Pagina inicial da secao")
    page_end: int = Field(..., description="Pagina final da secao")
    topic_name: str = Field(..., description="Titulo da secao")
    topic_number: str = Field(..., description="Identificador no documento")
    original_text: Optional[str] = Field(None, description="Conteudo associado")


class Index(BaseModel):
    topics: List[str]
    pages: List[int]


class FilteredDocs(BaseModel):
    items: List[IndexRefs]


class Extraction(BaseModel):
    text: str = Field(..., description="Texto extraido")


# LangGraph supervisor state -------------------------------------------------


class AgentState(TypedDict, total=False):
    """State propagated between supervisor and worker agents."""

    messages: Annotated[List[BaseMessage], add]
    next: str 