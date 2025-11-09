import os
import time
import fitz  # This is the PyMuPDF library
import chromadb
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---

# 1. SET YOUR FOLDER PATHS HERE
#    This assumes your script is in a "LegalProject" folder,
#    and the "ipc_sections", "cpc_sections", etc., are at the same level.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# List of single PDF files
"""
SINGLE_FILES = [
    os.path.join(BASE_DIR, "indian_constitution.pdf"),
    os.path.join(BASE_DIR, "ipc_sections", "ipc.pdf"), # Assuming the PDF is inside this folder
    os.path.join(BASE_DIR, "cpc_sections", "cpc.pdf")  # Assuming the PDF is inside this folder
]
"""

# List of directories containing judgment PDFs (2015-2025)
JUDGEMENT_DIRS = [
    os.path.join(BASE_DIR, "supreme_court_judgments", str(year))
    for year in range(2012, 2020) # 2015 to 2025
]

# 2. CONFIGURE YOUR LOCAL DATABASE
DB_PATH = os.path.join(BASE_DIR, "legal_db") # This will create a 'legal_db' folder

# 3. CONFIGURE CHUNKING
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# 4. CONFIGURE GEMINI API
#    (Remember to set the GEMINI_API_KEY environment variable!)
try:
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
except KeyError:
    print("="*50)
    print("ERROR: GEMINI_API_KEY not found in environment variables.")
    print("Please set it before running: set GEMINI_API_KEY=YOUR_KEY_HERE")
    print("="*50)
    exit()

# --- HELPER FUNCTIONS ---

def get_all_pdf_files():
    """Finds all PDF files in the configured paths."""
    pdf_files = []

    # Add all single files
    """
    for file_path in SINGLE_FILES:
        if os.path.exists(file_path):
            pdf_files.append(file_path)
        else:
            print(f"Warning: Single file not found at {file_path}")
    """
    # Add all files from judgment directories
    for dir_path in JUDGEMENT_DIRS:
        if os.path.exists(dir_path):
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.lower().endswith(".pdf"):
                        pdf_files.append(os.path.join(root, file))
        else:
            print(f"Warning: Directory not found at {dir_path}")
            
    return pdf_files

def extract_text_from_pdf(pdf_path):
    """Extracts all text from a single PDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def get_embedding_with_retry(text, model, retries=3, delay=5):
    """
    Gets embeddings from Gemini, with rate limit handling and retries.
    Your student plan likely has a "60 requests per minute" limit.
    """
    # Always wait 1 second to respect the 60 QPM limit
    time.sleep(1) 
    
    for attempt in range(retries):
        try:
            # Create the embedding
            embedding = genai.embed_content(
                model=model,
                content=text,
                task_type="RETRIEVAL_DOCUMENT" # Important for RAG
            )
            return embedding['embedding']
        except Exception as e:
            print(f"  > API Error: {e}. Retrying in {delay}s... (Attempt {attempt+1}/{retries})")
            time.sleep(delay)
    
    print(f"  > FAILED to get embedding for chunk after {retries} attempts.")
    return None

# --- MAIN SCRIPT ---

def main():
    print("Step 1: Building Knowledge Library...")
    
    # --- Part 1.1: Gather Documents ---
    all_pdfs = get_all_pdf_files()
    if not all_pdfs:
        print("No PDF files found. Please check your CONFIGURATION paths.")
        return
    
    print(f"Found {len(all_pdfs)} PDF documents to process.")
    
    # --- Part 1.2: Initialize Chunking ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # --- Part 1.4: Initialize Vector Database ---
    print(f"Initializing database at: {DB_PATH}")
    db_client = chromadb.PersistentClient(path=DB_PATH)
    collection = db_client.get_or_create_collection("legal_brain")
    
    # --- Process all files ---
    total_chunks_added = 0
    for pdf_path in all_pdfs:
        print(f"\nProcessing: {os.path.basename(pdf_path)}...")
        
        # 1.1: Read File
        document_text = extract_text_from_pdf(pdf_path)
        if not document_text:
            continue
            
        # 1.2: Chunk Document
        chunks = text_splitter.split_text(document_text)
        print(f"  > Split into {len(chunks)} chunks.")
        
        if len(chunks) == 0:
            print("  > No text found in document.")
            continue
            
        # 1.3 & 1.4: Embed and Store Chunks
        for i, chunk_text in enumerate(chunks):
            
            # Create a unique ID for this chunk
            chunk_id = f"{os.path.basename(pdf_path)}_{i}"
            
            # Check if this chunk ID already exists in the DB
            if collection.get(ids=[chunk_id])['ids']:
                print(f"  > Chunk {chunk_id} already in database. Skipping.")
                continue

            print(f"  > Creating embedding for chunk {i+1}/{len(chunks)}...")
            
            # 1.3: Embed (with retry logic)
            embedding = get_embedding_with_retry(
                text=chunk_text,
                model="models/text-embedding-004" # This is the one to use
            )
            
            if embedding:
                # 1.4: Store in ChromaDB
                relative_path = os.path.relpath(pdf_path, BASE_DIR).replace('\\', '/')
                
                collection.add(
                    embeddings=[embedding],
                    documents=[chunk_text],
                    metadatas=[{"source": relative_path}], # <-- UPDATED
                    ids=[chunk_id]
                )   
                total_chunks_added += 1
                print(f"  > Successfully added chunk {chunk_id}.")

    print("\n---" * 10)
    print("✅ Step 1 Complete! ✅")
    print(f"Successfully added {total_chunks_added} new chunks to the 'legal_brain' database.")
    print(f"Your local vector database is ready at: {DB_PATH}")

if __name__ == "__main__":
    main()