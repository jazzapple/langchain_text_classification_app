from typing import Literal

from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    category: Literal["World", "Sports", "Business", "Sci/Tech"] = Field(
        description="The news category that best describes the content. Valid values are World, Sports, Business, Sci/Tech"
    )
    reasoning: str = Field(
        description="A brief one-sentence explanation of why this category was chosen"
    )
