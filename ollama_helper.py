# Helper functions for interacting with Ollama LLM

import ollama
import os
import subprocess
import platform
import time

def check_ollama_installed():
    """
    Checks if Ollama is installed on the system
    
    Returns:
        bool: True if Ollama is installed, False otherwise
    """
    try:
        if platform.system() == "Windows":
            result = subprocess.run(["where", "ollama"], capture_output=True, text=True)
        else:  # Unix-like systems (Linux, macOS)
            result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
        
        return result.returncode == 0
    except Exception:
        return False

def check_ollama_running():
    """
    Checks if Ollama service is running
    
    Returns:
        bool: True if Ollama is running, False otherwise
    """
    try:
        # Try to list models - this will fail if Ollama is not running
        ollama.list()
        return True
    except Exception:
        return False

def start_ollama_service():
    """
    Attempts to start the Ollama service
    
    Returns:
        bool: True if service was started successfully, False otherwise
    """
    try:
        if platform.system() == "Windows":
            subprocess.Popen(["ollama", "serve"], 
                            creationflags=subprocess.CREATE_NO_WINDOW)
        else:  # Unix-like systems
            subprocess.Popen(["ollama", "serve"], 
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           start_new_session=True)
        
        # Wait for service to start
        time.sleep(5)
        return check_ollama_running()
    except Exception:
        return False

def check_model_exists(model_name):
    """
    Checks if a specific model is already downloaded
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        bool: True if model exists, False otherwise
    """
    try:
        models = ollama.list()
        return any(model['name'] == model_name for model in models.get('models', []))
    except Exception:
        return False

def pull_model(model_name):
    """
    Pulls (downloads) a model from Ollama library
    
    Args:
        model_name: Name of the model to pull
        
    Returns:
        bool: True if model was pulled successfully, False otherwise
    """
    try:
        ollama.pull(model_name)
        return True
    except Exception as e:
        print(f"Error pulling model: {str(e)}")
        return False

def generate_response(model_name, prompt, system_prompt=None, temperature=0.7, max_tokens=1000):
    """
    Generates a response from the LLM
    
    Args:
        model_name: Name of the model to use
        prompt: The user prompt
        system_prompt: Optional system prompt to set context
        temperature: Creativity parameter (0.0 to 1.0)
        max_tokens: Maximum tokens to generate
        
    Returns:
        str: The generated response
    """
    try:
        # Prepare parameters
        params = {
            'model': model_name,
            'prompt': prompt,
            'temperature': temperature,
            'num_predict': max_tokens,
        }
        
        # Add system prompt if provided
        if system_prompt:
            params['system'] = system_prompt
            
        # Generate response
        response = ollama.generate(**params)
        
        return response['response']
    except Exception as e:
        return f"Error generating response: {str(e)}"
