"""
Build Legal Knowledge Database using ChromaDB's Internal Embeddings
This creates a database compatible with both online (Gemini) and offline (Ollama) modes.
No API key required!
"""
import os
import time
import fitz  # PyMuPDF
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Single PDF files (Constitution, IPC, CPC)
SINGLE_FILES = [
    os.path.join(BASE_DIR, "indian_constitution.pdf"),
    os.path.join(BASE_DIR, "ipc_sections", "ipc.pdf"),
    os.path.join(BASE_DIR, "cpc_sections", "cpc.pdf")
]

# Supreme Court judgments directories (2012-2019)
JUDGEMENT_DIRS = [
    os.path.join(BASE_DIR, "supreme_court_judgments", str(year))
    for year in range(2012, 2020)
]

# New database path (separate from Gemini database)
DB_PATH = os.path.join(BASE_DIR, "legal_db_chroma")

# Chunking configuration
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

print("="*70)
print("  Legal Knowledge Database Builder (ChromaDB Embeddings)")
print("="*70)
print(f"\nDatabase will be created at: {DB_PATH}")
print("This database uses ChromaDB's internal embeddings (no API key needed)")
print("Compatible with both online and offline modes!\n")

# --- HELPER FUNCTIONS ---

def get_all_pdf_files():
    """Finds all PDF files in the configured paths."""
    pdf_files = []
    
    # Add single files
    for file_path in SINGLE_FILES:
        if os.path.exists(file_path):
            pdf_files.append(file_path)
            print(f"[FOUND] {os.path.basename(file_path)}")
        else:
            print(f"[SKIP] Not found: {file_path}")
    
    # Add judgment files
    for dir_path in JUDGEMENT_DIRS:
        if os.path.exists(dir_path):
            count = 0
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.lower().endswith(".pdf"):
                        pdf_files.append(os.path.join(root, file))
                        count += 1
            print(f"[FOUND] {count} judgments in {os.path.basename(dir_path)}")
        else:
            print(f"[SKIP] Directory not found: {dir_path}")
    
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
        print(f"[ERROR] Could not read {pdf_path}: {e}")
        return None

# --- MAIN SCRIPT ---

def main():
    print("\n" + "="*70)
    print("STEP 1: Gathering Documents")
    print("="*70)
    
    all_pdfs = get_all_pdf_files()
    if not all_pdfs:
        print("\n[ERROR] No PDF files found. Please check your paths.")
        return
    
    print(f"\n[OK] Found {len(all_pdfs)} PDF documents to process")
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    # Initialize ChromaDB with default embeddings
    print("\n" + "="*70)
    print("STEP 2: Initializing ChromaDB")
    print("="*70)
    print(f"Creating database at: {DB_PATH}")
    print("Using ChromaDB's internal sentence-transformer embeddings...")
    
    db_client = chromadb.PersistentClient(path=DB_PATH)
    
    # Create or get collection (will use default embeddings)
    try:
        collection = db_client.get_collection("legal_brain_chroma")
        print(f"[OK] Collection 'legal_brain_chroma' already exists")
        print(f"[INFO] Current chunks in database: {collection.count()}")
    except:
        collection = db_client.create_collection("legal_brain_chroma")
        print(f"[OK] Created new collection 'legal_brain_chroma'")
    
    # Process all files
    print("\n" + "="*70)
    print("STEP 3: Processing Documents")
    print("="*70)
    
    total_chunks_added = 0
    total_chunks_skipped = 0
    
    for idx, pdf_path in enumerate(all_pdfs, 1):
        filename = os.path.basename(pdf_path)
        print(f"\n[{idx}/{len(all_pdfs)}] Processing: {filename}")
        
        # Read PDF
        document_text = extract_text_from_pdf(pdf_path)
        if not document_text:
            print(f"  [SKIP] Could not extract text")
            continue
        
        print(f"  [OK] Extracted {len(document_text)} characters")
        
        # Chunk document
        chunks = text_splitter.split_text(document_text)
        print(f"  [OK] Split into {len(chunks)} chunks")
        
        if len(chunks) == 0:
            print(f"  [SKIP] No text found")
            continue
        
        # Add chunks to database
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{filename}_{i}"
            
            # Check if already exists
            try:
                existing = collection.get(ids=[chunk_id])
                if existing['ids']:
                    total_chunks_skipped += 1
                    continue
            except:
                pass
            
            # Add to database (ChromaDB will auto-generate embeddings)
            try:
                relative_path = os.path.relpath(pdf_path, BASE_DIR).replace('\\', '/')
                
                collection.add(
                    documents=[chunk_text],
                    metadatas=[{"source": relative_path}],
                    ids=[chunk_id]
                )
                total_chunks_added += 1
                
                # Progress indicator
                if total_chunks_added % 10 == 0:
                    print(f"  [PROGRESS] Added {total_chunks_added} chunks...")
                
            except Exception as e:
                print(f"  [ERROR] Could not add chunk {i}: {e}")
        
        print(f"  [DONE] Added {len(chunks)} chunks from this document")
    
    # Final summary
    print("\n" + "="*70)
    print("BUILD COMPLETE!")
    print("="*70)
    print(f"Total chunks added: {total_chunks_added}")
    print(f"Total chunks skipped (already existed): {total_chunks_skipped}")
    print(f"Total chunks in database: {collection.count()}")
    print(f"\nDatabase location: {DB_PATH}")
    print("\nThis database uses ChromaDB's internal embeddings (384 dimensions)")
    print("Compatible with both online and offline modes!")
    print("\nNext step: Update app.py to use this database")
    print("="*70)

if __name__ == "__main__":
    main()
