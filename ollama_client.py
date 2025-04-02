import os
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
import httpx

logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Client for interacting with the Ollama API.
    
    This class provides methods to check server status, list available models,
    and generate text using either the generate or chat API endpoints.
    """
    
    def __init__(self, base_url: Optional[str] = None, timeout: int = 10):
        """
        Initialize the Ollama client.
        
        Args:
            base_url: URL of the Ollama API server (default: from env var or localhost)
            timeout: Timeout for API requests in seconds
        """
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.timeout = timeout
        logger.info(f"Initialized Ollama client with base URL: {self.base_url}")
    
    async def check_server(self) -> bool:
        """
        Check if the Ollama server is available.
        
        Returns:
            True if server is available, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server: {str(e)}")
            return False
    
    async def list_models(self) -> List[str]:
        """
        Get a list of available models from the Ollama server.
        
        Returns:
            List of model names
            
        Raises:
            ConnectionError: If the server is not available
            RuntimeError: If the API request fails
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                
                if response.status_code != 200:
                    error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                model_data = response.json()
                return [model.get("name") for model in model_data.get("models", [])]
        except httpx.RequestError as e:
            logger.error(f"Ollama API request error: {str(e)}")
            raise ConnectionError(f"Failed to connect to Ollama server: {str(e)}")
    
    async def generate(
        self, 
        model: str, 
        prompt: str, 
        system: str = "", 
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate text using the Ollama generate API.
        
        Args:
            model: Name of the model to use
            prompt: Input text prompt
            system: System prompt for context
            parameters: Additional parameters for the model (temperature, etc.)
            
        Returns:
            Generated text response
            
        Raises:
            ConnectionError: If the server is not available
            RuntimeError: If the API request fails
        """
        try:
            url = f"{self.base_url}/api/generate"
            
            payload = {
                "model": model,
                "prompt": prompt,
                "system": system,
                "stream": False
            }
            
            # Add optional parameters if provided
            if parameters:
                payload.update(parameters)
            
            logger.debug(f"Sending generate request to {url} with model {model}")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                
                if response.status_code != 200:
                    error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                result = response.json()
                return result.get("response", "").strip()
                
        except httpx.RequestError as e:
            logger.error(f"Ollama API request error: {str(e)}")
            raise ConnectionError(f"Failed to connect to Ollama server: {str(e)}")
    
    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate text using the Ollama chat API.
        
        Args:
            model: Name of the model to use
            messages: List of message objects with role and content
            parameters: Additional parameters for the model (temperature, etc.)
            
        Returns:
            Generated chat response
            
        Raises:
            ConnectionError: If the server is not available
            RuntimeError: If the API request fails
        """
        try:
            url = f"{self.base_url}/api/chat"
            
            payload = {
                "model": model,
                "messages": messages,
                "stream": False
            }
            
            # Add optional parameters if provided
            if parameters:
                payload["options"] = parameters
            
            logger.debug(f"Sending chat request to {url} with model {model}")
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                
                if response.status_code != 200:
                    error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                result = response.json()
                return result.get("message", {}).get("content", "").strip()
                
        except httpx.RequestError as e:
            logger.error(f"Ollama API request error: {str(e)}")
            raise ConnectionError(f"Failed to connect to Ollama server: {str(e)}")
    
    async def generate_with_system(
        self,
        model: str,
        prompt: str,
        system: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Helper method to generate text with a system prompt.
        Automatically chooses between generate and chat APIs based on model.
        
        Args:
            model: Name of the model to use
            prompt: Input text prompt
            system: System prompt for context
            parameters: Additional parameters for the model (temperature, etc.)
            
        Returns:
            Generated text response
        """
        # Check if OLLAMA_USE_CHAT_API environment variable is set
        use_chat_api = os.getenv("OLLAMA_USE_CHAT_API", "").lower() in ("true", "1", "yes")
        
        # If not set, use a simple heuristic based on model name
        if not use_chat_api:
            use_chat_api = any(name in model.lower() for name in ["llama3", "mistral", "gemma", "gpt"])
        
        if use_chat_api:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
            return await self.chat(model, messages, parameters)
        else:
            return await self.generate(model, prompt, system, parameters)

# Create a singleton instance for easy import
default_client = OllamaClient()

# Convenience function to get client with specified base_url
def get_client(base_url: Optional[str] = None) -> OllamaClient:
    """Get an Ollama client with the specified base_url"""
    if base_url:
        return OllamaClient(base_url=base_url)
    return default_client 