from pydantic import BaseModel, Field
from typing import List

class NewsletterThemeOutput(BaseModel):
    """Output model for structured theme and sub-theme generation."""
    theme: str = Field(description="The main newsletter theme based on the provided article titles.")
    sub_themes: List[str] = Field(description="List of sub-themes or key news items to investigate under the main theme.")