from typing import Literal

from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    category: Literal["World", "Sports", "Business", "Sci/Tech"] = Field(
        description="The news category that best describes the content. Valid values are World, Sports, Business, Sci/Tech"
    )
    reasoning: str = Field(
        description="A brief one-sentence explanation of why this category was chosen"
    )


class CategoryProbabilities(BaseModel):
    world: float = Field(description="Probability this article belongs to World (0.0–1.0)")
    sports: float = Field(description="Probability this article belongs to Sports (0.0–1.0)")
    business: float = Field(description="Probability this article belongs to Business (0.0–1.0)")
    sci_tech: float = Field(description="Probability this article belongs to Sci/Tech (0.0–1.0)")
