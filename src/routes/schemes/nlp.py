from pydantic import BaseModel
from typing import List, Optional

class PushRequest(BaseModel):
    do_reset: Optional[int] = 0

class SearchRequest(BaseModel):
    text: str
    limit: Optional[int] = 5

class TranslationRequest(BaseModel):
    text: str
    target_language: str

class SummaryRequest(BaseModel):
    chunk_ids: List[int]
    max_length: int = 500