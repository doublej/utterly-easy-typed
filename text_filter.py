import re
import os
import asyncio
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Union, Any
from enum import Enum, auto
import logging
import httpx
from ollama_client import get_client, OllamaClient

# Configure logging but use NullHandler by default
# (actual handlers will be configured by main.py)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.handlers = []  # Remove default handlers
logger.addHandler(logging.NullHandler())  # Add null handler to suppress output


class FilterError(Exception):
    """Base exception for text filter errors"""
    pass


class LLMProcessingError(FilterError):
    """Exception raised when LLM processing fails"""
    pass


class FilterValidationError(FilterError):
    """Exception raised when filter validation fails"""
    pass


class TextFilterMode(ABC):
    """
    Abstract base class for text filter modes.
    Each filter mode implements a specific way to transform input text.
    """
    name: str
    description: str

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.log_message = None  # Will be set by TextFilterManager

    def set_logger(self, log_func):
        """Set the logger function to use for messages"""
        self.log_message = log_func

    @abstractmethod
    async def process(self, text: str) -> str:
        """
        Process the input text according to the filter mode's rules.
        
        Args:
            text: The input text to be processed
            
        Returns:
            Processed text
            
        Raises:
            FilterError: If processing fails
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"


class RawFilterMode(TextFilterMode):
    """Filter mode that returns text with no modifications"""
    
    def __init__(self):
        super().__init__(
            name="raw",
            description="No text modifications"
        )
    
    async def process(self, text: str) -> str:
        return text


class CodeFilterMode(TextFilterMode):
    """Filter mode that converts text to snake_case for code"""
    
    def __init__(self):
        super().__init__(
            name="code",
            description="Convert text to snake_case for code"
        )
    
    async def process(self, text: str) -> str:
        try:
            # Convert spaces and remove special characters
            result = re.sub(r'[^\w\s]', '', text.lower())
            # Replace spaces with underscores
            result = re.sub(r'\s+', '_', result)
            # Remove consecutive underscores
            result = re.sub(r'_+', '_', result)
            # Remove leading/trailing underscores
            result = result.strip('_')
            return result
        except Exception as e:
            if self.log_message:
                self.log_message(f"Error in code filter mode: {str(e)}")
            else:
                logger.error(f"Error in code filter mode: {str(e)}")
            raise FilterError(f"Failed to process text in code mode: {str(e)}")


class CleanFilterMode(TextFilterMode):
    """Filter mode that converts text to lowercase and removes special characters"""
    
    def __init__(self):
        super().__init__(
            name="clean",
            description="Lowercase text and remove special characters"
        )
    
    async def process(self, text: str) -> str:
        try:
            # Convert to lowercase
            result = text.lower()
            # Remove special characters
            result = re.sub(r'[^\w\s]', '', result)
            # Normalize whitespace
            result = re.sub(r'\s+', ' ', result)
            # Trim whitespace
            result = result.strip()
            return result
        except Exception as e:
            if self.log_message:
                self.log_message(f"Error in clean filter mode: {str(e)}")
            else:
                logger.error(f"Error in clean filter mode: {str(e)}")
            raise FilterError(f"Failed to process text in clean mode: {str(e)}")


class OllamaFilterMode(TextFilterMode):
    """
    Filter mode that uses Ollama for text transformations.
    
    This mode sends text to a local Ollama instance for processing
    with a specified model and prompt template.
    """
    
    def __init__(
        self, 
        name: str, 
        description: str,
        model: str,
        system_prompt: str,
        base_url: Optional[str] = None,
        timeout: int = 10,
        use_chat_api: bool = False,
        parameters: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name=name, description=description)
        # First check environment variables, then fall back to provided values
        self.model = os.getenv("OLLAMA_MODEL", model)
        self.system_prompt = system_prompt
        self.base_url = os.getenv("OLLAMA_BASE_URL", base_url or "http://localhost:11434")
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT", str(timeout)))
        self.use_chat_api = os.getenv("OLLAMA_USE_CHAT_API", "").lower() in ("true", "1", "yes") or use_chat_api
        self.parameters = parameters or {}
        # Create a client instance with the current settings
        self.client = get_client(base_url=self.base_url)
    
    async def process(self, text: str) -> str:
        """
        Process text using Ollama API.
        
        Args:
            text: Input text to process
            
        Returns:
            Processed text from Ollama
            
        Raises:
            LLMProcessingError: If Ollama API call fails
        """
        try:
            if self.use_chat_api:
                # Use the chat API endpoint
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ]
                result = await self.client.chat(
                    model=self.model,
                    messages=messages,
                    parameters=self.parameters
                )
            else:
                # Use the generate API endpoint
                result = await self.client.generate(
                    model=self.model,
                    prompt=text,
                    system=self.system_prompt,
                    parameters=self.parameters
                )
            
            return result.strip()
                
        except Exception as e:
            if self.log_message:
                self.log_message(f"Error in Ollama filter mode: {str(e)}")
            else:
                logger.error(f"Error in Ollama filter mode: {str(e)}")
            raise LLMProcessingError(f"Failed to process text with Ollama: {str(e)}")

#
# class CombinedFilterMode(TextFilterMode):
#     """
#     Filter mode that combines multiple filter modes in sequence.
#
#     This allows for chaining transformations, such as an LLM transformation
#     followed by rule-based cleanup.
#     """
#
#     def __init__(
#         self,
#         name: str,
#         description: str,
#         filters: List[TextFilterMode]
#     ):
#         super().__init__(name=name, description=description)
#         if not filters:
#             raise ValueError("CombinedFilterMode requires at least one filter")
#         self.filters = filters
#
#     async def process(self, text: str) -> str:
#         """
#         Process text through each filter in sequence.
#
#         Args:
#             text: Input text to process
#
#         Returns:
#             Text processed through all filters
#
#         Raises:
#             FilterError: If any filter processing fails
#         """
#         result = text
#         for filter_mode in self.filters:
#             try:
#                 result = await filter_mode.process(result)
#             except Exception as e:
#                 if self.log_message:
#                     self.log_message(f"Error in combined filter mode with {filter_mode.name}: {str(e)}")
#                 else:
#                     logger.error(f"Error in combined filter mode with {filter_mode.name}: {str(e)}")
#                 raise FilterError(f"Failed in combined mode at {filter_mode.name}: {str(e)}")
#         return result


class TextFilterManager:
    """
    Manages multiple text filter modes and handles switching between them.
    """
    
    def __init__(self):
        self.filters: Dict[str, TextFilterMode] = {}
        self.current_filter: Optional[str] = None
        self.log_message = None  # Will be set by transcriber
        
        # Register built-in filters
        self.register_filter(RawFilterMode())
        self.register_filter(CodeFilterMode())
        self.register_filter(CleanFilterMode())
        
        # Register LLM-based filters
        try:
            # Register code formatter filter (converts natural language to code)
            code_formatter = create_code_formatter_filter()
            self.register_filter(code_formatter)
            
            # Register grammar correction filter
            grammar_filter = create_grammar_correction_filter()
            self.register_filter(grammar_filter)


            # Register grammar correction filter
            helper_filter = create_helper_correction_filter()
            self.register_filter(helper_filter)


            if self.log_message:
                self.log_message("Successfully registered LLM-based filters")
            else:
                logger.info("Successfully registered LLM-based filters")
        except Exception as e:
            if self.log_message:
                self.log_message(f"Failed to register LLM-based filters: {str(e)}")
            else:
                logger.warning(f"Failed to register LLM-based filters: {str(e)}")
        
        # Set default filter
        if self.filters:
            self.current_filter = "raw"
    
    def set_logger(self, log_func):
        """Set the logger function to use for messages"""
        self.log_message = log_func
        # Also set logger for all filter modes
        for filter_mode in self.filters.values():
            filter_mode.set_logger(log_func)
    
    def register_filter(self, filter_mode: TextFilterMode) -> None:
        """
        Register a new filter mode.
        
        Args:
            filter_mode: The filter mode to register
            
        Raises:
            ValueError: If a filter with the same name already exists
        """
        if filter_mode.name in self.filters:
            raise ValueError(f"Filter '{filter_mode.name}' already exists")
        
        # If we have a logger, set it for the filter mode
        if self.log_message:
            filter_mode.set_logger(self.log_message)
            
        self.filters[filter_mode.name] = filter_mode
        if self.log_message:
            self.log_message(f"Registered filter mode: {filter_mode.name}")
        else:
            logger.info(f"Registered filter mode: {filter_mode.name}")
    
    def unregister_filter(self, filter_name: str) -> None:
        """
        Unregister a filter mode.
        
        Args:
            filter_name: Name of the filter to unregister
            
        Raises:
            ValueError: If the filter doesn't exist or is the current filter
        """
        if filter_name not in self.filters:
            raise ValueError(f"Filter '{filter_name}' does not exist")
        
        if self.current_filter == filter_name:
            raise ValueError(f"Cannot unregister current filter '{filter_name}'")
        
        del self.filters[filter_name]
        if self.log_message:
            self.log_message(f"Unregistered filter mode: {filter_name}")
        else:
            logger.info(f"Unregistered filter mode: {filter_name}")
    
    def set_filter(self, filter_name: str) -> None:
        """
        Set the current filter mode.
        
        Args:
            filter_name: Name of the filter to set as current
            
        Raises:
            ValueError: If the filter doesn't exist
        """
        if filter_name not in self.filters:
            raise ValueError(f"Filter '{filter_name}' does not exist")
        
        self.current_filter = filter_name
        if self.log_message:
            self.log_message(f"Set current filter mode to: {filter_name}")
        else:
            logger.info(f"Set current filter mode to: {filter_name}")
    
    def get_current_filter(self) -> TextFilterMode:
        """
        Get the current filter mode.
        
        Returns:
            The current TextFilterMode
            
        Raises:
            ValueError: If no current filter is set
        """
        if not self.current_filter:
            raise ValueError("No current filter is set")
        
        return self.filters[self.current_filter]
    
    def get_available_filters(self) -> Dict[str, str]:
        """
        Get all available filter modes with descriptions.
        
        Returns:
            Dictionary of filter names and descriptions
        """
        return {name: filter_mode.description for name, filter_mode in self.filters.items()}
    
    async def process_text(self, text: str, filter_name: Optional[str] = None) -> str:
        """
        Process text using either the specified filter or the current filter.
        
        Args:
            text: Input text to process
            filter_name: Optional name of filter to use instead of current
            
        Returns:
            Processed text
            
        Raises:
            ValueError: If the specified filter doesn't exist or no current filter
            FilterError: If processing fails
        """
        if filter_name:
            if filter_name not in self.filters:
                raise ValueError(f"Filter '{filter_name}' does not exist")
            filter_mode = self.filters[filter_name]
        else:
            filter_mode = self.get_current_filter()
        
        try:
            return await filter_mode.process(text)
        except Exception as e:
            if self.log_message:
                self.log_message(f"Error processing text with filter '{filter_mode.name}': {str(e)}")
            else:
                logger.error(f"Error processing text with filter '{filter_mode.name}': {str(e)}")
            raise FilterError(f"Failed to process text with filter '{filter_mode.name}': {str(e)}")


# Example of creating an Ollama-based filter
def create_code_formatter_filter(base_url: Optional[str] = None) -> OllamaFilterMode:
    """Create an Ollama-based filter for formatting code from natural language"""
    model = os.getenv("OLLAMA_CODE_MODEL", "codellama:7b-instruct")
    
    return OllamaFilterMode(
        name="code_formatter",
        description="Converts natural language description to working code",
        model=model,
        system_prompt=(
            "You are a helpful assistant that converts natural language descriptions "
            "into clean, working code. Extract the programming intent from the input "
            "and output only valid, executable code without any explanation or comments."
        ),
        base_url=base_url,
        parameters={
            "temperature": float(os.getenv("OLLAMA_CODE_TEMP", "0.1")),
            "top_p": float(os.getenv("OLLAMA_CODE_TOP_P", "0.9")),
            "top_k": int(os.getenv("OLLAMA_CODE_TOP_K", "40"))
        }
    )


def create_grammar_correction_filter(base_url: Optional[str] = None) -> OllamaFilterMode:
    """Create an Ollama-based filter for grammar correction"""
    model = os.getenv("OLLAMA_GRAMMAR_MODEL", "llama3:8b")
    
    return OllamaFilterMode(
        name="grammar",
        description="Corrects grammar and improves text readability",
        model=model,
        system_prompt=(
            "You are a professional editor. Fix grammar, spelling, and punctuation "
            "in the provided text. Maintain the original meaning and tone. "
            "Return only the corrected text without any explanations or comments."
        ),
        base_url=base_url,
        use_chat_api=os.getenv("OLLAMA_USE_CHAT_API", "").lower() in ("true", "1", "yes"),
        parameters={
            "temperature": float(os.getenv("OLLAMA_GRAMMAR_TEMP", "0.3"))
        }
    )


def create_helper_correction_filter(base_url: Optional[str] = None) -> OllamaFilterMode:
    """Create an Ollama-based filter for """
    model = os.getenv("OLLAMA_HELPER_MODEL", "llama3:8b")

    return OllamaFilterMode(
        name="helper",
        description="Understands intent and smooths formatting",
        model=model,
        system_prompt=(
            "You are a transcription helper. You recognize transcriber output and find misrecognized words and correct it. You recognize oddly placed punctiuation and correct it. You recognize token issues with the model like random 'thank you for watching' and remove it"
            "crucial: Return only the corrected text without any explanations or comments."
            "<examples>"
            "<in>"
            "Alright. Thank you. Focus on the, ehm... So I need you to... Performance of the... Transcription. So... Is it properly? Accepting. Bye. ... silences in between the sentences. Ok. Thanks for watching!"
            "</in>"
            "<out>"
            "Thanks. Please focus on the perforance of the transcription. Is it properly accepting the silences in between the sentences?"
            "</out>"
            "<in>"
            "Regular expression that matches. Sentences that start with... Code. Thank you for watching! thank thank thank"
            "</in>"
            "<out>"
            "A regular expression that matches sentences that start with 'code'"
            "</out>"
            "</examples>"

        ),
        base_url=base_url,
        use_chat_api=os.getenv("OLLAMA_USE_CHAT_API", "true").lower() in ("true", "1", "yes"),
        parameters={
            "temperature": float(os.getenv("OLLAMA_HELPER_TEMP", "0.1"))
        }
    )

