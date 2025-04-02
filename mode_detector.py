import os
import logging
import re
from typing import Optional, List, Dict, Any, Tuple, Set
from ollama_client import get_client, OllamaClient

# Configure logging but use NullHandler by default
# (actual handlers will be configured by main.py)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.handlers = []  # Remove default handlers
logger.addHandler(logging.NullHandler())  # Add null handler to suppress output

class ModeDetectionError(Exception):
    """Exception raised when mode detection fails"""
    pass

class ModeDetector:
    """
    Detects voice commands for mode switching using Ollama.
    
    This class analyzes transcribed text to determine if it contains
    a command to switch between different text filter modes.
    """
    
    def __init__(
        self,
        min_text_length: int = 4,
        max_text_length: int = 50,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 10
    ):
        """
        Initialize the mode detector.
        
        Args:
            min_text_length: Minimum text length to consider for mode detection
            max_text_length: Maximum text length to consider for mode detection
            model: Ollama model to use for detection (default from env var)
            base_url: Ollama API URL (default from env var or localhost)
            timeout: Request timeout in seconds
        """
        # Load configuration from environment variables or use defaults
        self.min_text_length = int(os.getenv('UTTERTYPE_MODE_MIN_LENGTH', min_text_length))
        self.max_text_length = int(os.getenv('UTTERTYPE_MODE_MAX_LENGTH', max_text_length))
        self.model = os.getenv('UTTERTYPE_MODE_MODEL', model or 'tinyllama')
        
        # Get Ollama URL from environment variables
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.log_message = None  # Will be set by transcriber
        self._log(f"ModeDetector initialized with Ollama URL: {self.base_url}")
        
        # Initialize Ollama client with explicit base_url
        self.client = get_client(base_url=self.base_url)
        
        # Valid modes - moved here as a class attribute for easier access
        self.valid_modes = ["raw", "code", "clean", "grammar", "code_formatter", "helper"]
        
        # Command patterns that indicate an explicit mode switch intent
        self.command_patterns = [
            r"switch\s+to\s+(\w+)",
            r"change\s+to\s+(\w+)",
            r"use\s+(\w+)\s+mode",
            r"set\s+mode\s+to\s+(\w+)",
            r"go\s+to\s+(\w+)",
            r"switch\s+(\w+)\s+mode"
        ]

        # Create system prompt template
        self.system_prompt = """
            You are a voice command detector specialized in identifying mode switch commands.
            Your task is to determine if the transcribed text contains a command to switch to a specific mode.
            
            Available modes:
            - raw: No text modifications
            - code: Convert text to snake_case for code
            - clean: Lowercase text and remove special characters
            - grammar: Fix grammar and spelling
            - code_formatter: Format code snippets properly
            - helper: fixes issues with transcribed text
            
            Rules:
            1. Analyze ONLY the command intent, not the actual content
            2. Respond with ONLY the mode name (raw, code, clean, grammar, code_formatter) if detected
            3. Respond with "none" if no clear mode switch command is detected
            4. Ignore any content other than the mode switching command
            5. Commands might include phrases like "switch to", "change to", "use mode", etc.
            
            Be forgiving about phrasing and recognize casual requests.
        """
    
    def set_logger(self, log_func):
        """Set the logger function to use for messages"""
        self.log_message = log_func
    
    def _log(self, message, level="info"):
        """Log a message either via log_message or the standard logger"""
        if self.log_message:
            self.log_message(message)
        else:
            if level == "info":
                logger.info(message)
            elif level == "error":
                logger.error(message)
            elif level == "warning":
                logger.warning(message)
        
    async def detect_mode_switch(self, text: str) -> Optional[str]:
        """
        Detect if text contains a command to switch to a specific mode.
        
        Args:
            text: Transcribed text to analyze
            
        Returns:
            Mode name if detected, None otherwise
            
        Raises:
            ModeDetectionError: If mode detection fails
        """
        # Skip processing if text is too short or too long
        if len(text.strip()) < self.min_text_length:
            return None
            
        # Skip processing if text is too long (likely not a command)
        if len(text.strip()) > self.max_text_length:
            self._log(f"Text too long for mode detection: {len(text.strip())} chars")
            return None
            
        # Lowered text for easier processing
        lower_text = text.lower().strip()
        
        # First try to detect mode command patterns directly to avoid unnecessary LLM calls
        direct_match, mode_matches = self._check_direct_patterns(lower_text)
        
        # If we have one clear direct match, return it immediately without using LLM
        if direct_match:
            self._log(f"Detected direct mode switch command: {direct_match}")
            return direct_match
            
        # If we find multiple mode names in the text without clear command patterns,
        # it's likely not a mode switch command (e.g., "clean up this code")
        if len(mode_matches) > 1:
            self._log(f"Multiple mode mentions without clear command pattern: {mode_matches}")
            return None
            
        # If no direct matches but one mode is mentioned, or we're not sure,
        # use the LLM for more sophisticated detection
        try:
            # Check if Ollama server is available
            server_available = await self.client.check_server()
            if not server_available:
                self._log(f"Ollama server not available at {self.base_url}", "error")
                raise ModeDetectionError(f"Ollama server not available at {self.base_url}")
            
            # Prepare the prompt for Ollama
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text}
            ]
            
            # Get response from Ollama
            response = await self.client.chat(
                model=self.model,
                messages=messages
            )
            
            # Parse and validate the response
            mode = self._parse_mode_response(response)
            return mode
            
        except Exception as e:
            self._log(f"Mode detection error: {str(e)}", "error")
            raise ModeDetectionError(f"Failed to detect mode: {str(e)}")
    
    def _check_direct_patterns(self, text: str) -> Tuple[Optional[str], Set[str]]:
        """
        Check text for direct command patterns and mode mentions.
        
        Args:
            text: Normalized text to check
            
        Returns:
            Tuple of (direct match if found, set of all mode names mentioned)
        """
        mode_matches = set()
        direct_match = None
        
        # Check for specific command patterns
        for pattern in self.command_patterns:
            matches = re.search(pattern, text)
            if matches:
                # Extract the potential mode from the regex capture group
                potential_mode = matches.group(1).strip().lower()
                if potential_mode in self.valid_modes:
                    direct_match = potential_mode
                    break
        
        # Find all mode names mentioned in the text
        for mode in self.valid_modes:
            if mode in text:
                mode_matches.add(mode)
                
        return direct_match, mode_matches
    
    def _parse_mode_response(self, response: str) -> Optional[str]:
        """
        Parse the response from Ollama and extract the mode.
        
        Args:
            response: Text response from Ollama
            
        Returns:
            Mode name if valid, None otherwise
        """
        # Clean up the response
        response = response.strip().lower()
        
        # Check if response is a valid mode
        if response in self.valid_modes:
            return response
            
        # If response contains "none" or no specific mode was detected
        if "none" in response:
            return None
            
        # Try to extract a valid mode from the response text
        for mode in self.valid_modes:
            if mode in response:
                return mode
                
        return None 