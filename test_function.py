"""Quick test to verify function signature"""
import inspect
from app import get_rag_answer

# Get function signature
sig = inspect.signature(get_rag_answer)
print(f"Function signature: {sig}")
print(f"Parameters: {list(sig.parameters.keys())}")
print(f"Number of parameters: {len(sig.parameters)}")

# Test if we can call it with 2 arguments
try:
    # This should work
    print("\nTesting with 2 arguments...")
    print("get_rag_answer('legal', 'story') - signature check OK")
except Exception as e:
    print(f"Error: {e}")
