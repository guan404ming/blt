"""Shared models for translation framework"""

from typing import Optional
from pydantic import BaseModel, Field


class MusicConstraints(BaseModel):
    """Music constraints for lyrics translation"""

    syllable_counts: list[int] = Field(description="Target syllable count per line")
    rhyme_scheme: Optional[str] = Field(
        default=None, description="Rhyme scheme (e.g., AABB, ABAB, AAAA)"
    )
    syllable_patterns: Optional[list[list[int]]] = Field(
        default=None,
        description="Target syllable patterns per line (e.g., [[1,1,1,3], [1,3,2,4]])",
    )
