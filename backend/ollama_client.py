"""
Ollama Client for Tunisie Telecom Q&A System

This module provides a client for interacting with the Ollama API,
allowing the backend to use locally hosted models for generating responses.
"""

import requests
import hashlib
import logging
import time
from typing import List, Dict, Optional, Any, Union

# Default configuration (will be imported from main backend)
class Config:
    """Default configuration for Ollama client"""
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llama3"
    OLLAMA_TEMPERATURE = 0.7
    OLLAMA_MAX_TOKENS = 1024
    OLLAMA_ENABLED = True
    OLLAMA_CACHE_SIZE = 100
    OLLAMA_CACHE_TTL = 3600  # 1 hour

class OllamaClient:
    """Enhanced client for interacting with Ollama API with caching and better error handling"""
    
    def __init__(self, base_url=Config.OLLAMA_URL, model=Config.OLLAMA_MODEL):
        """
        Initialize Ollama client with base URL and model

        Args:
            base_url (str): Base URL of the Ollama API
            model (str): Default model to use
        """
        self.base_url = base_url
        self.model = model
        self.api_endpoint = f"{base_url}/api/generate"
        self.logger = logging.getLogger("TT_QA.OllamaClient")
        self.logger.info(f"Ollama client initialized with model: {model}")
        
        # Response cache to avoid redundant API calls
        self.response_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = Config.OLLAMA_CACHE_SIZE
        self.cache_ttl = Config.OLLAMA_CACHE_TTL
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available models from Ollama
        
        Returns:
            List[str]: List of model names available in Ollama
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)  # Add timeout
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            else:
                self.logger.error(f"Failed to get models. Status: {response.status_code}")
                return []
        except requests.exceptions.ConnectionError:
            self.logger.error("Connection error: Ollama server is not running or unreachable")
            return []
        except requests.exceptions.Timeout:
            self.logger.error("Timeout error: Ollama server took too long to respond")
            return []
        except Exception as e:
            self.logger.error(f"Error connecting to Ollama: {str(e)}")
            return []
    
    def is_available(self) -> bool:
        """
        Check if Ollama service is available
        
        Returns:
            bool: True if Ollama is available, False otherwise
        """
        try:
            # Try a faster health check first
            response = requests.get(f"{self.base_url}/api/health", timeout=2)
            return response.status_code == 200
        except:
            try:
                # Fall back to checking models if health endpoint is not available
                models = self.get_available_models()
                return len(models) > 0
            except:
                return False
    
    def _generate_cache_key(self, prompt: str, system_prompt: Optional[str], model: str, temperature: float) -> str:
        """
        Generate a unique cache key based on request parameters
        
        Args:
            prompt (str): The prompt text
            system_prompt (Optional[str]): The system prompt
            model (str): The model name
            temperature (float): The temperature setting
            
        Returns:
            str: MD5 hash to use as cache key
        """
        key_components = [prompt, system_prompt or "", model, str(temperature)]
        return hashlib.md5("||".join(key_components).encode()).hexdigest()
    
    def _cleanup_cache(self) -> None:
        """Clean up expired cache entries"""
        now = time.time()
        expired_keys = [
            k for k, v in self.response_cache.items() 
            if now > v.get("expiry", 0)
        ]
        
        for key in expired_keys:
            del self.response_cache[key]
            
    def generate(self, 
                prompt: str, 
                system_prompt: Optional[str] = None, 
                temperature: float = Config.OLLAMA_TEMPERATURE, 
                max_tokens: int = Config.OLLAMA_MAX_TOKENS) -> str:
        """
        Generate response using Ollama API with caching
        
        Args:
            prompt (str): The prompt to send to Ollama
            system_prompt (Optional[str]): System prompt for context
            temperature (float): Temperature for generation
            max_tokens (int): Maximum tokens to generate
            
        Returns:
            str: The generated response
        """
        # Clean up expired cache entries
        self._cleanup_cache()
        
        # Check if response is in cache
        cache_key = self._generate_cache_key(prompt, system_prompt, self.model, temperature)
        
        if cache_key in self.response_cache:
            self.cache_hits += 1
            self.logger.debug(f"Cache hit for query. Cache hits: {self.cache_hits}")
            return self.response_cache[cache_key]["response"]
            
        self.cache_misses += 1
        self.logger.debug(f"Cache miss for query. Cache misses: {self.cache_misses}")
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            # Add system prompt if provided
            if system_prompt:
                payload["system"] = system_prompt
                
            self.logger.debug(f"Sending request to Ollama: {self.api_endpoint}")
            response = requests.post(self.api_endpoint, json=payload, timeout=30)  # Add timeout
            
            if response.status_code == 200:
                data = response.json()
                result = data.get("response", "No response from Ollama")
                
                # Cache the response
                if len(self.response_cache) >= self.max_cache_size:
                    # Remove oldest item if cache is full (based on timestamp)
                    oldest_key = min(
                        self.response_cache.items(), 
                        key=lambda x: x[1].get("timestamp", 0)
                    )[0]
                    del self.response_cache[oldest_key]
                
                self.response_cache[cache_key] = {
                    "response": result,
                    "timestamp": time.time(),
                    "expiry": time.time() + self.cache_ttl
                }
                return result
            else:
                error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                self.logger.error(error_msg)
                return f"Error: Failed to get response from Ollama (Status: {response.status_code})"
                
        except requests.exceptions.ConnectionError:
            error_msg = "Connection error: Ollama server is not running or unreachable"
            self.logger.error(error_msg)
            return error_msg
        except requests.exceptions.Timeout:
            error_msg = "Timeout error: Ollama server took too long to respond"
            self.logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error calling Ollama API: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
            
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a specific model
        
        Args:
            model_name (Optional[str]): Name of the model to get info about. Uses current model if None.
            
        Returns:
            Dict: Model information or empty dict if not found
        """
        model = model_name or self.model
        try:
            response = requests.get(f"{self.base_url}/api/show?name={model}", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Failed to get model info. Status: {response.status_code}")
                return {}
        except Exception as e:
            self.logger.error(f"Error getting model info: {str(e)}")
            return {}
            
    def pull_model(self, model_name: str) -> Dict[str, Any]:
        """
        Pull a model from Ollama repository
        
        Args:
            model_name (str): Name of the model to pull
            
        Returns:
            Dict: Result of the pull operation
        """
        try:
            self.logger.info(f"Pulling model {model_name} from Ollama repository")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=60  # Longer timeout for model downloads
            )
            
            if response.status_code == 200:
                return {"success": True, "message": f"Model {model_name} pulled successfully"}
            else:
                error_msg = f"Failed to pull model {model_name}. Status: {response.status_code}"
                self.logger.error(error_msg)
                return {"success": False, "message": error_msg}
        except Exception as e:
            error_msg = f"Error pulling model {model_name}: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "message": error_msg}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Ollama client
        
        Returns:
            Dict: Statistics about cache usage and current model
        """
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": len(self.response_cache),
            "current_model": self.model,
            "is_available": self.is_available()
        }
