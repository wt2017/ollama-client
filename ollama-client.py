import requests
import json
import socket
from typing import Generator

class OllamaClient:
    def __init__(self, host: str, port: int):
        # Remove http:// prefix if it exists in the host parameter
        if host.startswith("http://"):
            host = host[7:]
        elif host.startswith("https://"):
            host = host[8:]
        self.base_url = f"http://{host}:{port}/api"
    
    def list_models(self):
        """List all available models"""
        url = f"{self.base_url}/tags"
        response = requests.get(url)
        print(f"Status code for list_models: {response.status_code}")
        if response.status_code == 200:
            return response.json().get("models", [])
        else:
            print(f"Error listing models: {response.text}")
            return []
        
    def generate(self, model: str, prompt: str, stream: bool = False, **kwargs) -> dict:
        """非流式调用"""
        url = f"{self.base_url}/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }
        response = requests.post(url, json=data)
        
        # Print the status code and response text for debugging
        print(f"Status code: {response.status_code}")
        print(f"Response text: {response.text[:500]}...")  # Show first 500 chars
        
        return response.json()
    
    def generate_stream(self, model: str, prompt: str, **kwargs) -> Generator[str, None, None]:
        """流式调用"""
        url = f"{self.base_url}/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            **kwargs
        }
        with requests.post(url, json=data, stream=True) as response:
            for chunk in response.iter_lines():
                if chunk:
                    yield json.loads(chunk.decode())["response"]

if __name__ == "__main__":
    # Check if the hostname is resolvable
    hostname = "hzllmapi.dyn.nesc.nokia.net"
    try:
        # Try to resolve the hostname first
        ip = socket.gethostbyname(hostname)
        print(f"Hostname resolves to: {ip}")
        
        # 使用示例
        client = OllamaClient(hostname, 8080)
        
        # 先列出所有可用模型
        print("Getting available models...")
        available_models = client.list_models()
        
        if not available_models:
            print("No models available or couldn't retrieve model list.")
            exit(1)
            
        print("Available models:")
        for model in available_models:
            print(f"- {model['name']} (tags: {model.get('tags', [])})")
            
        # 使用第一个可用的模型
        model_name = available_models[0]['name']
        print(f"\nUsing model: {model_name}")
        
        # 非流式调用
        result = client.generate(
            model=model_name,
            prompt="为什么天空是蓝色的?",
            options={"temperature": 0.7}
        )
        # Check if 'response' key exists in the result
        if "response" in result:
            print("完整响应:", result["response"])
        else:
            print("API response format is unexpected. Full response:", result)
    
        # 流式调用
        print("流式响应:")
        for chunk in client.generate_stream(
            model=model_name,
            prompt="请用中文自我介绍"
        ):
            print(chunk, end="", flush=True)
    except socket.gaierror as e:
        print(f"Error resolving hostname: {hostname}")
        print(f"Error details: {e}")
        print("Please check if the hostname is correct or try using an IP address directly")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your connection and server settings")
