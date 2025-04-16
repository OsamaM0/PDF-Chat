from fastapi import UploadFile, HTTPException
import difflib
import io
import tempfile
import os
import logging
import uuid
from typing import Dict, Any, Optional, List

class VoiceController:
    def __init__(self, generation_client):
        # self.model = whisper.load_model("base")
        self.generation_client = generation_client
        self.audio_storage: Dict[str, bytes] = {}
        logging.basicConfig(level=logging.ERROR)

    def text_to_speech(self, text: str) -> Optional[io.BytesIO]:
        """Converts text to speech using gTTS and returns audio stream."""
        try:
            audio_stream = self.generation_client.text_to_speech(text)
            return audio_stream
        except Exception as e:
            logging.exception(f"Error in TTS: {e}")
            return None

    def compare_texts(self, text1: str, text2: str) -> List[Dict[str, Any]]:
        """Compares two texts and returns detailed changes."""
        seqm = difflib.SequenceMatcher(None, text1.split(), text2.split())
        changes = []
        for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
            if opcode == 'equal':
                continue
            elif opcode == 'insert':
                changes.append({"type": "added", "text": " ".join(text2.split()[b0:b1])})
            elif opcode == 'delete':
                changes.append({"type": "removed", "text": " ".join(text1.split()[a0:a1])})
            elif opcode == 'replace':
                changes.append({
                    "type": "replaced",
                    "original": " ".join(text2.split()[b0:b1]),
                    "replacement": " ".join(text1.split()[a0:a1])
                })
        return changes

    async def process_transcription(
        self,
        file: UploadFile,
        expected_text: str,
        language: str = 'en',
        prompt: str = ""
    ) -> Dict[str, Any]:
        temp_audio_path = None
        try:
            if not file.content_type.startswith("audio/"):
                raise HTTPException(status_code=400, detail="Invalid file type. Only audio files are accepted.")

            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_audio_file:
                file_content = await file.read()
                temp_audio_file.write(file_content)
                temp_audio_path = temp_audio_file.name

            # Transcribe audio file
            transcription_request = {
                "audio_filepath": temp_audio_path,
                "prompt": prompt,
                "language": language
            }
            result = self.generation_client.transcribe(**transcription_request)

            transcribed_text = result.strip()
            differences = self.compare_texts(transcribed_text, expected_text)

            response_data = {
                "transcribed_text": transcribed_text,
                "expected_text": expected_text,
                "changes": [],
                "confidence_score": None  # Add confidence score if needed
            }

            # Generate TTS for replacement changes
            for change in differences:
                change_data = {"type": change["type"]}

                if change["type"] == "replaced":
                    change_data["original"] = change["original"]
                    change_data["replacement"] = change["replacement"]

                    # Generate TTS for replacement text
                    replacement_audio = self.text_to_speech(change["original"])
                    if replacement_audio:
                        unique_id = str(uuid.uuid4())
                        replacement_audio_path = f"assets/audio_changes/{unique_id}_replacement.mp3"
                        os.makedirs(os.path.dirname(replacement_audio_path), exist_ok=True)
                        # with open(replacement_audio_path, "wb") as f:
                        #     f.write(replacement_audio.read())
                        replacement_audio.stream_to_file(replacement_audio_path)
                        change_data["replacement_audio_url"] = f"/audio/{unique_id}_replacement.mp3"

                response_data["changes"].append(change_data)

            return response_data
        finally:
            if temp_audio_path:
                try:
                    os.remove(temp_audio_path)
                except (FileNotFoundError, Exception) as e:
                    logging.error(f"Error removing temporary file: {e}")



    def clear_storage(self) -> None:
        """Clear the audio storage."""
        self.audio_storage.clear()