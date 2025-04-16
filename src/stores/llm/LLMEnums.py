from enum import Enum

class LLMEnums(Enum):
    OPENAI = "OPENAI"
    COHERE = "COHERE"
    G4F = "G4F"

class OpenAIEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    
    # STT = "whisper-1"
    STT = "gpt-4o-mini-transcribe"
    TTS = "gpt-4o-mini-tts"
    VOICE = "alloy"
    

class CoHereEnums(Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "CHATBOT"

    DOCUMENT = "search_document"
    QUERY = "search_query"

class G4FEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    

class DocumentTypeEnum(Enum):
    DOCUMENT = "document"
    QUERY = "query"