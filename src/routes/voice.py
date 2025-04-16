import os
import uuid
from fastapi import APIRouter, Depends, File, Request, UploadFile, BackgroundTasks, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import io
from typing import Dict, Any, Optional
from controllers.VoiceController import VoiceController
from .schemes.voice import (
    TextToSpeechRequest,
    TextToSpeechResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    AudioResponse,
    ErrorResponse
)

voice_router = APIRouter(
    prefix="/api/v1/voice",
    tags=["Speech Processing"],
)

@voice_router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def transcribe_audio(
    request: Request,
    audio_file: UploadFile = File(..., description="The audio file to transcribe", alias="audio_file"),
    expected_text: str = Form(..., description="The expected text to compare against"),
    language: Optional[str] = Form(default="en", description="Language of the audio")
) -> TranscriptionResponse:
    """
    Transcribe audio file and compare with expected text.
    Returns transcription results with detailed changes and confidence scores.
    """
    try:
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an audio file."
            )
        
        voice_controller = VoiceController(generation_client=request.app.generation_client)

        response_data = await voice_controller.process_transcription(
            file=audio_file,
            expected_text=expected_text,
            language=language
        )
        return TranscriptionResponse(**response_data)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@voice_router.post(
    "/text-to-speech",
    response_model=TextToSpeechResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def text_to_speech(request: Request,  tts_request: TextToSpeechRequest) -> TextToSpeechResponse:
    """
    Convert text to speech.
    Returns URL to retrieve the generated audio file.
    """
    voice_controller = VoiceController(generation_client=request.app.generation_client)
    
    audio_stream = voice_controller.text_to_speech(
        text=tts_request.text,
    )
    
    if audio_stream:
        unique_id = str(uuid.uuid4())
        replacement_audio_path = f"assets/audio_changes/{unique_id}_tts.mp3"
        os.makedirs(os.path.dirname(replacement_audio_path), exist_ok=True)
        audio_stream.stream_to_file(replacement_audio_path)
        
        
        return TextToSpeechResponse(
            audio_url=f"{unique_id}_tts.mp3",
        )

    else:
        raise HTTPException(status_code=500, detail="Failed to generate speech")


@voice_router.get(
    "/audio/{audio_id}",
    response_class=StreamingResponse,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def get_audio(audio_id: str) -> StreamingResponse:
    """
    Retrieve audio file by ID.
    Returns the audio file as a streaming response.
    """
    try:
        base_path = os.path.abspath("assets/audio_changes")
        audio_path = os.path.join(base_path, f"{audio_id}.mp3")
        print(f"Accessing file at: {audio_path}")

        if not os.path.exists(audio_path):
            raise HTTPException(status_code=404, detail="Audio file not found")
        if os.stat(audio_path).st_size == 0:
            raise HTTPException(status_code=500, detail="Audio file is empty or corrupted")

        return FileResponse(audio_path)

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
