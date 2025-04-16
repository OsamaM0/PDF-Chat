from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class TextToSpeechRequest(BaseModel):
    text: str = Field(..., description="The text to convert to speech", min_length=1)

class TextToSpeechResponse(BaseModel):
    audio_url: str = Field(..., description="URL to retrieve the generated audio")

class TranscriptionRequest(BaseModel):
    expected_text: str = Field(..., description="The expected text to compare against", min_length=1)
    language: str = Field(default="en", description="The language of the audio")
    
class ChangeDetail(BaseModel):
    type: str = Field(..., description="Type of change: added, removed, or replaced")
    text: Optional[str] = Field(None, description="The text content of the change")
    original: Optional[str] = Field(None, description="Original text for replaced type")
    replacement: Optional[str] = Field(None, description="Replacement text for replaced type")
    replacement_audio_url: Optional[str] = Field(None, description="URL for the audio of replacement text")
    added_audio_url: Optional[str] = Field(None, description="URL for the audio of added text")

class TranscriptionResponse(BaseModel):
    transcribed_text: str = Field(..., description="The text transcribed from the audio")
    expected_text: str = Field(..., description="The original expected text")
    changes: List[ChangeDetail] = Field(list(), description="List of differences between texts")
    confidence_score: Optional[float] = Field(None, description="Confidence score of the transcription")

class AudioResponse(BaseModel):
    audio_id: str = Field(..., description="Unique identifier for the audio file")
    duration: float = Field(..., description="Duration of the audio in seconds")
    format: str = Field(..., description="Format of the audio file (e.g., 'mp3')")
    size: int = Field(..., description="Size of the audio file in bytes")

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code for specific error types")