import io
from typing import Optional, List
import logging
import g4f
from ..LLMInterface import LLMInterface
from ..LLMEnums import G4FEnums, DocumentTypeEnum
import numpy as np

# Attempt to import SentenceTransformer with a fallback mechanism
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    logging.getLogger(__name__).error(
        "Failed to import SentenceTransformer. This might be due to compatibility issues. "
        "Try running fix_dependencies.sh or fix_dependencies.cmd to resolve this."
    )
    SENTENCE_TRANSFORMER_AVAILABLE = False

class G4FProvider(LLMInterface):

    def __init__(self,
                default_input_max_characters: int=1000,
                default_generation_max_output_tokens: int=1000,
                default_generation_temperature: float=0.1):
        
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model = None
        self.embedding_size = None

        # Initialize g4f settings
        g4f.debug.logging = False  # Set to True for debugging
        
        self.enums = G4FEnums
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        """Set the generation model to use"""
        self.generation_model_id = model_id
        self.logger.info(f"Set generation model to: {model_id}")

    def set_embedding_model(self, model_id: str, embedding_size: int):
        """Initialize the sentence transformer embedding model"""
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            self.logger.error("SentenceTransformer is not available. Embedding functionality will not work.")
            return

        try:
            self.embedding_model = SentenceTransformer(model_id)
            self.embedding_size = embedding_size
            self.logger.info(f"Initialized embedding model: {model_id} with size {embedding_size}")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            raise

    def process_text(self, text: str) -> str:
        """Process input text by truncating if necessary"""
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int=None,
                      temperature: float=None) -> Optional[str]:
        """Generate text using G4F providers"""
        if not self.generation_model_id:
            self.logger.error("Generation model for G4F was not set")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        try:
            # Convert chat history to format expected by G4F
            messages = []
            
            # First add any history messages
            for msg in chat_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Then add the current prompt as a user message if not already in history
            if not messages or messages[-1]["role"] != G4FEnums.USER.value or messages[-1]["content"] != prompt:
                messages.append({
                    "role": G4FEnums.USER.value,
                    "content": self.process_text(prompt)
                })

            # Get all available providers dynamically from g4f
            all_providers = []
            
            # Prioritize OpenAI-based providers first if they exist
            if hasattr(g4f.Provider, 'OpenaiChat'):
                all_providers.append(g4f.Provider.OpenaiChat)
            
            # Try to add other common providers
            for provider_name in dir(g4f.Provider):
                # Skip special attributes, already added providers, and non-provider items
                if (provider_name.startswith('_') or 
                    provider_name == 'OpenaiChat' or
                    not hasattr(g4f.Provider, provider_name)):
                    continue
                
                provider = getattr(g4f.Provider, provider_name)
                # Check if it's a proper provider class (has necessary attributes)
                if hasattr(provider, '__name__'):
                    all_providers.append(provider)
            
            # Add None as the last fallback (auto-select provider)
            if all_providers:
                self.logger.info(f"Found {len(all_providers)} available G4F providers")
            else:
                self.logger.warning("No specific G4F providers found, will use auto-select only")
            all_providers.append(None)
            
            response = None
            last_error = None
            
            for provider in all_providers:
                try:
                    provider_name = provider.__name__ if provider else 'auto'
                    self.logger.info(f"Trying provider: {provider_name}")
                    response = g4f.ChatCompletion.create(
                        model=self.generation_model_id,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_output_tokens,
                        provider=provider
                    )
                    
                    if response and len(response) > 10:  # Ensure we got a meaningful response
                        self.logger.info(f"Successfully got response from provider: {provider_name}")
                        break
                    else:
                        self.logger.warning(f"Provider {provider_name} returned too short response: {response}")
                except Exception as e:
                    last_error = e
                    self.logger.warning(f"Provider {provider_name if provider else 'auto'} failed: {e}")
                    continue
            
            if not response:
                self.logger.error(f"All G4F providers failed. Last error: {last_error}")
                return "I apologize, but I'm unable to generate a response at the moment. Please try again later."
            
            return response
        
        except Exception as e:
            self.logger.error(f"Error while generating text with G4F: {e}")
            return "Sorry, an error occurred while processing your request."

    def embed_text(self, text: str, document_type: str = None) -> List[float]:
        """Generate embeddings using Sentence Transformers"""
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            self.logger.error("SentenceTransformer is not available. Using fallback random embeddings.")
            # Return a random vector of the correct size as a fallback
            if self.embedding_size:
                return list(np.random.rand(self.embedding_size))
            return None
            
        if not self.embedding_model:
            self.logger.error("Embedding model was not set")
            return None
        
        try:
            # Generate embeddings
            embedding = self.embedding_model.encode(self.process_text(text))
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Error while creating embeddings: {e}")
            return None

    def construct_prompt(self, prompt: str, role: str) -> dict:
        """Construct a message in the format expected by the LLM"""
        return {
            "role": role,
            "content": self.process_text(prompt)
        }
    
    def text_to_speech(self, text: str) -> Optional[io.BytesIO]:
        """
        This is a placeholder for TTS functionality
        Currently not implemented in G4F
        """
        self.logger.warning("Text-to-speech is not implemented for G4F provider")
        return None
    
    def transcribe(self, audio_filepath: str, prompt: str, language: str = "en") -> str:
        """
        This is a placeholder for STT functionality
        Currently not implemented in G4F
        """
        self.logger.warning("Transcription is not implemented for G4F provider")
        return f"Transcription not available in G4F provider. Text content: {prompt if prompt else '[No text provided]'}"
