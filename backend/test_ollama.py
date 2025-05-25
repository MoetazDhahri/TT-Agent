#!/usr/bin/env python3
"""
Test script for Ollama integration for Tunisie Telecom Q&A System
"""

import argparse
import sys
import json
import os  # For path manipulation
# Ensure the script's directory is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pprint import pprint
from ollama_client import OllamaClient

def test_ollama_client():
    """Test basic Ollama client functionality"""
    print("=== Testing Ollama Client ===")
    
    # Create client
    client = OllamaClient()
    print(f"Client initialized with model: {client.model}")
    
    # Check if Ollama is available
    print("\nChecking if Ollama is available...")
    is_available = client.is_available()
    print(f"Ollama available: {is_available}")
    
    if not is_available:
        print("Error: Ollama is not running. Please start the Ollama server.")
        return False
    
    # Get available models
    print("\nGetting available models...")
    models = client.get_available_models()
    print(f"Available models: {models}")
    
    if not models:
        print("No models found. You might need to pull a model first.")
        return False
    
    # Test text generation
    print("\nTesting text generation...")
    
    try:
        test_prompt = "What services does Tunisie Telecom offer?"
        system_prompt = "You are a helpful assistant for Tunisie Telecom."
        
        print(f"Prompt: {test_prompt}")
        print(f"System prompt: {system_prompt}")
        
        response = client.generate(test_prompt, system_prompt)
        print("\nResponse:")
        print("-" * 40)
        print(response)
        print("-" * 40)
        
        # Test caching
        print("\nTesting cache (same query)...")
        cached_response = client.generate(test_prompt, system_prompt)
        cache_stats = client.get_stats()
        print(f"Cache stats: {cache_stats}")
        
        return True
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        return False

def test_model_switching(model_name=None):
    """Test switching between models"""
    print("\n=== Testing Model Switching ===")
    
    client = OllamaClient()
    
    # Get available models
    models = client.get_available_models()
    if len(models) < 2:
        print("Need at least 2 models for switching test. Skipping.")
        return True
    
    # Use provided model or select a different model
    current_model = client.model
    if model_name:
        test_model = model_name
    else:
        for model in models:
            if model != current_model:
                test_model = model
                break
        else:
            print("Could not find an alternative model. Skipping.")
            return True
    
    print(f"Switching from {current_model} to {test_model}")
    
    # Switch model
    client.model = test_model
    
    # Test generation with new model
    test_prompt = "Hello, please introduce yourself as an AI assistant."
    response = client.generate(test_prompt)
    
    print(f"Response from {test_model}:")
    print("-" * 40)
    print(response[:200] + "..." if len(response) > 200 else response)
    print("-" * 40)
    
    # Switch back
    client.model = current_model
    print(f"Switched back to {current_model}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Test Ollama integration")
    parser.add_argument("--model", help="Specify a model to test")
    args = parser.parse_args()
    
    # Run the tests
    if not test_ollama_client():
        sys.exit(1)
    
    if args.model:
        test_model_switching(args.model)
    else:
        test_model_switching()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()
