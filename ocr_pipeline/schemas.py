"""
Pydantic models for data validation and API responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Union

class ErrorTroubleshooting(BaseModel):
    name: str
    description: str
    troubleshooting: str

class OCRResult(BaseModel):
    raw_text: str = Field(..., description="Raw text extracted by OCR engine")
    clean_text: Optional[str] = Field(None, description="Cleaned and validated text")
    confidence: float = Field(0.0, description="Confidence score of the extraction")
    error_details: Optional[Union[ErrorTroubleshooting, List[ErrorTroubleshooting]]] = Field(
        None, description="Matched error code details"
    )
    is_valid: bool = Field(False, description="Whether the extracted text matches a known error code")
