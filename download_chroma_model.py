"""
Pre-download ChromaDB embedding model to avoid timeout during first query.
Run this once before starting the app.
"""
import chromadb
import os

print("="*60)
print("ChromaDB Model Downloader")
print("="*60)
print("\nThis will download the all-MiniLM-L6-v2 model (~79MB)")
print("This only needs to be done once.\n")

# Get the database path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "legal_db")

print(f"Database path: {DB_PATH}")
print("\nConnecting to ChromaDB...")

try:
    # Connect to database
    db_client = chromadb.PersistentClient(path=DB_PATH)
    collection = db_client.get_collection("legal_brain")
    
    print("✓ Connected to database")
    print("\nTriggering model download by running a test query...")
    
    # Run a simple query to trigger model download
    results = collection.query(
        query_texts=["test query"],
        n_results=1
    )
    
    print("\n" + "="*60)
    print("✓ SUCCESS! Model downloaded and cached.")
    print("="*60)
    print("\nYou can now run your app without timeout issues:")
    print("  python app.py")
    print("\n")
    
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    print("\nMake sure you have:")
    print("1. Run build_database.py first")
    print("2. The legal_db folder exists")
    print("3. Internet connection for model download")
