import llama_cpp
import os
import time
import requests

MODEL_PATH = os.path.join(os.path.dirname(__file__), "offline-access", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Loader animation
class Loader:
    def __init__(self, message="Loading..."):
        self.message = message
        self.running = False
        self.current_char = 0
    def start(self):
        self.running = True
        print(self.message, end=" ", flush=True)
    def update(self):
        chars = "|/-\\"
        print(chars[self.current_char % 4], end="\r", flush=True)
        self.current_char += 1
        time.sleep(0.2)
    def stop(self):
        self.running = False
        print("Done!")

def check_ollama():
    """Check if Ollama is running and query it."""
    print("\n" + "="*60)
    print("Testing Ollama (GPU-Accelerated Option)")
    print("="*60)
    loader = Loader("Checking Ollama availability...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            print("✅ Ollama is running!")
            
            # Test query
            print("\nRunning test inference with Ollama...")
            loader = Loader("Inferencing with Ollama...")
            
            response = requests.post(
                OLLAMA_API_URL,
                json={
                    "model": "mistral",
                    "prompt": "What is the capital of France?",
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n✅ Ollama Response:\n{result.get('response', '')}")
                print("\n✅ Ollama with GPU support is working perfectly!")
                return True
            else:
                print(f"❌ Ollama API error: {response.status_code}")
                return False
        else:
            print("❌ Ollama API returned error")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Ollama is not running!")
        print("   To use Ollama, download from https://ollama.ai and run: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False
    finally:
        loader.stop()

def check_gpu_model_usage():
    """Check Mistral model with GPU support."""
    print("\n" + "="*60)
    print("Testing Mistral Model (Local llama-cpp-python)")
    print("="*60)
    loader = Loader("Loading Mistral model...")
    try:
        model = llama_cpp.Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=-1,  # Offload all layers to GPU
            verbose=True
        )
        loader.stop()
        
        sys_info = llama_cpp.llama_print_system_info().decode()
        print("\nSystem Info:\n", sys_info)
        
        # Run a quick inference to check GPU usage
        loader = Loader("Running inference...")
        prompt = "What is the capital of France?"
        output = model.create_completion(prompt, max_tokens=32)
        loader.stop()
        
        print("\nModel Output:\n", output["choices"][0]["text"])
        print("\n✅ If you see CUDA/cuBLAS and your GPU listed above, GPU is being used.")
    except Exception as e:
        loader.stop()
        print(f"❌ Error loading model or using GPU: {e}")
    finally:
        pass

if __name__ == "__main__":
    print("\n" + "="*60)
    print("GPU & Model Configuration Checker")
    print("="*60)
    
    # Check Ollama first (preferred)
    ollama_ok = check_ollama()
    
    # Check Mistral as fallback
    check_gpu_model_usage()
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    if ollama_ok:
        print("✅ Ollama is available with GPU support - RECOMMENDED")
    else:
        print("⚠️  Ollama not running - Will use Mistral model as fallback")
    print("="*60 + "\n")
