import requests
import json
import socket
import sys
import base64
import argparse
import cv2
import os
import tempfile
from typing import Generator, List, Dict, Any

class OllamaClient:
    def __init__(self, host: str, port: int):
        # Remove http:// prefix if it exists in the host parameter
        if host.startswith("http://"):
            host = host[7:]
        elif host.startswith("https://"):
            host = host[8:]
        self.base_url = f"http://{host}:{port}/api"

    def extract_video_frames(self, video_path: str, max_frames: int = 5) -> List[str]:
        """Extract frames from a video file and convert them to base64"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {video_path}")
            
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video info: {total_frames} frames, {fps} fps, {duration:.2f} seconds")
        
        # Calculate frame intervals to extract evenly distributed frames
        if max_frames >= total_frames:
            interval = 1
            max_frames = total_frames
        else:
            interval = total_frames // max_frames
            
        frame_base64_list = []
        
        for i in range(max_frames):
            # Set position to the next frame to capture
            frame_pos = i * interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            
            # Read the frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert frame to base64
            success, buffer = cv2.imencode(".jpg", frame)
            if success:
                base64_frame = base64.b64encode(buffer).decode('utf-8')
                frame_base64_list.append(base64_frame)
                print(f"Extracted frame {i+1}/{max_frames} at position {frame_pos}")
        
        cap.release()
        return frame_base64_list
        
    def process_video(self, model: str, video_path: str, max_frames: int = 5) -> Dict[str, Any]:
        """Process a video file and generate descriptions using the vision-language model"""
        try:
            # Extract frames from the video
            frames = self.extract_video_frames(video_path, max_frames)
            
            if not frames:
                return {"error": "No frames could be extracted from the video"}
                
            # Create a prompt with all frames
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please describe what is happening in this video in detail."}
                    ]
                }
            ]
            
            # Add each frame as image content
            for i, frame in enumerate(frames):
                messages[0]["content"].append({
                    "type": "image",
                    "image": frame
                })
                
            # Generate description using the model
            response = self.generate(
                model=model,
                prompt=json.dumps(messages),
                options={"temperature": 0.7}
            )
            
            return response
            
        except Exception as e:
            return {"error": str(e)}
    
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
    # Setup command-line argument parser
    parser = argparse.ArgumentParser(description='Process video with Ollama Vision-Language Model')
    parser.add_argument('--host', type=str, default='hzllmapi.dyn.nesc.nokia.net', help='Ollama server hostname or IP')
    parser.add_argument('--port', type=int, default=8080, help='Ollama server port')
    parser.add_argument('--model', type=str, default='qwen2.5vl:32b', help='Model name to use (default: qwen2.5vl:32b)')
    parser.add_argument('--video', type=str, help='Path to the video file to process')
    parser.add_argument('--frames', type=int, default=5, help='Number of frames to extract from video (default: 5)')
    parser.add_argument('--text-prompt', type=str, help='Text prompt for non-video requests')
    parser.add_argument('--output', type=str, help='Path to output file to save the response text')
    args = parser.parse_args()

    # Check if the hostname is resolvable
    hostname = args.host
    try:
        # Try to resolve the hostname first
        ip = socket.gethostbyname(hostname)
        print(f"Hostname resolves to: {ip}")
        
        # Create client
        client = OllamaClient(hostname, args.port)
        
        # Verify if the requested model exists
        print("Getting available models...")
        available_models = client.list_models()
        
        if not available_models:
            print("No models available or couldn't retrieve model list.")
            exit(1)
            
        # Find if our model exists in the available models
        model_names = [model['name'] for model in available_models]
        print(f"Available models: {', '.join(model_names)}")
        
        # Check if specified model exists, otherwise use qwen2.5vl:32b or fall back to first available
        model_name = args.model
        if model_name not in model_names:
            if "qwen2.5vl:32b" in model_names:
                model_name = "qwen2.5vl:32b"
                print(f"Specified model not found, using '{model_name}' instead.")
            else:
                model_name = model_names[0]
                print(f"Specified model not found and qwen2.5vl:32b not available, using '{model_name}' as fallback.")
            
        print(f"\nUsing model: {model_name}")
        
        # Check if we have a video to process
        if args.video:
            print(f"Processing video: {args.video}")
            result = client.process_video(
                model=model_name,
                video_path=args.video,
                max_frames=args.frames
            )
            
            # Check if 'response' key exists in the result
            if "response" in result:
                output_text = result["response"]
                print("\nVideo description:")
                print(output_text)
                
                # Save to output file if specified
                if args.output:
                    try:
                        with open(args.output, 'w', encoding='utf-8') as f:
                            f.write(output_text)
                        print(f"\nOutput saved to: {args.output}")
                    except Exception as e:
                        print(f"Error saving to output file: {e}")
            else:
                print("API response format is unexpected. Full response:", result)
        
        # If no video but we have a text prompt
        elif args.text_prompt:
            print(f"Generating response for text prompt: {args.text_prompt}")
            result = client.generate(
                model=model_name,
                prompt=args.text_prompt,
                options={"temperature": 0.7}
            )
            
            # Check if 'response' key exists in the result
            if "response" in result:
                output_text = result["response"]
                print("\nResponse:")
                print(output_text)
                
                # Save to output file if specified
                if args.output:
                    try:
                        with open(args.output, 'w', encoding='utf-8') as f:
                            f.write(output_text)
                        print(f"\nOutput saved to: {args.output}")
                    except Exception as e:
                        print(f"Error saving to output file: {e}")
            else:
                print("API response format is unexpected. Full response:", result)
        
        # If no video or text prompt, show usage
        else:
            print("No video or text prompt provided. Use --video or --text-prompt to specify input.")
            print("Example: python ollama-client.py --video my_video.mp4")
            print("Example: python ollama-client.py --text-prompt \"What is the capital of France?\"")
            print("Example: python ollama-client.py --video my_video.mp4 --output result.txt")
            
    except socket.gaierror as e:
        print(f"Error resolving hostname: {hostname}")
        print(f"Error details: {e}")
        print("Please check if the hostname is correct or try using an IP address directly")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your connection and server settings")
