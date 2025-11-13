"""
Universal Legal Knowledge Database Builder
Uses ChromaDB's built-in sentence-transformer embeddings (all-MiniLM-L6-v2)
Compatible with BOTH online (Gemini) and offline (Ollama) modes
No API key required!

This creates a NEW database separate from the existing Gemini-based one.
"""
import os
import sys
import fitz  # PyMuPDF
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime

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

# NEW Universal database path (separate from existing databases)
DB_PATH = os.path.join(BASE_DIR, "legal_db_universal")

# Chunking configuration
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

print("="*80)
print("  UNIVERSAL Legal Knowledge Database Builder")
print("="*80)
print(f"\n📦 Database Location: {DB_PATH}")
print("🔧 Embedding Model: ChromaDB built-in (all-MiniLM-L6-v2, 384 dimensions)")
print("✅ Compatible with: Gemini API (online) + Ollama/Mistral (offline)")
print("🔑 API Key Required: NO - completely universal!")
print("\n" + "="*80 + "\n")

# --- HELPER FUNCTIONS ---

def get_all_pdf_files():
    """Finds all PDF files in the configured paths."""
    pdf_files = []
    
    print("📂 Scanning for PDF documents...")
    print("-" * 80)
    
    # Add single files
    print("\n1️⃣  Core Legal Documents:")
    for file_path in SINGLE_FILES:
        if os.path.exists(file_path):
            pdf_files.append(file_path)
            filename = os.path.basename(file_path)
            print(f"   ✅ {filename}")
        else:
            filename = os.path.basename(file_path)
            print(f"   ❌ {filename} - NOT FOUND")
    
    # Add judgment files
    print("\n2️⃣  Supreme Court Judgments:")
    total_judgments = 0
    for dir_path in JUDGEMENT_DIRS:
        year = os.path.basename(dir_path)
        if os.path.exists(dir_path):
            count = 0
            for root, _, files in os.walk(dir_path):
                for file in files:
                    if file.lower().endswith(".pdf"):
                        pdf_files.append(os.path.join(root, file))
                        count += 1
            total_judgments += count
            print(f"   ✅ {year}: {count} judgments")
        else:
            print(f"   ❌ {year}: Directory not found")
    
    print(f"\n📊 Total documents found: {len(pdf_files)}")
    print(f"   - Core documents: {len([f for f in pdf_files if any(s in f for s in ['constitution', 'ipc', 'cpc'])])}")
    print(f"   - Judgments: {total_judgments}")
    print("-" * 80)
    
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
        return None

# --- MAIN SCRIPT ---

def main():
    start_time = datetime.now()
    
    # Step 1: Gather documents
    print("\n" + "="*80)
    print("STEP 1: GATHERING DOCUMENTS")
    print("="*80)
    
    all_pdfs = get_all_pdf_files()
    if not all_pdfs:
        print("\n❌ ERROR: No PDF files found. Please check your paths.")
        print("\nRequired files:")
        print("  - indian_constitution.pdf")
        print("  - ipc_sections/ipc.pdf")
        print("  - cpc_sections/cpc.pdf")
        print("  - supreme_court_judgments/2012-2019/*.pdf")
        return
    
    # Step 2: Initialize ChromaDB
    print("\n" + "="*80)
    print("STEP 2: INITIALIZING CHROMADB")
    print("="*80)
    print(f"\n📍 Database path: {DB_PATH}")
    print("🔧 Initializing ChromaDB with built-in embeddings...")
    
    try:
        db_client = chromadb.PersistentClient(path=DB_PATH)
        print("✅ ChromaDB client initialized")
    except Exception as e:
        print(f"❌ ERROR: Could not initialize ChromaDB: {e}")
        return
    
    # Create or get collection
    collection_name = "legal_brain_universal"
    try:
        collection = db_client.get_collection(collection_name)
        existing_count = collection.count()
        print(f"✅ Collection '{collection_name}' already exists")
        print(f"📊 Current chunks in database: {existing_count}")
        
        response = input("\n⚠️  Database exists. Continue adding new documents? (y/n): ")
        if response.lower() != 'y':
            print("❌ Aborted by user")
            return
    except:
        collection = db_client.create_collection(collection_name)
        print(f"✅ Created new collection '{collection_name}'")
        print("📊 Starting with empty database")
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    print(f"✅ Text splitter initialized (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    
    # Step 3: Process documents
    print("\n" + "="*80)
    print("STEP 3: PROCESSING DOCUMENTS")
    print("="*80)
    
    total_chunks_added = 0
    total_chunks_skipped = 0
    total_docs_processed = 0
    total_docs_failed = 0
    
    for idx, pdf_path in enumerate(all_pdfs, 1):
        filename = os.path.basename(pdf_path)
        relative_path = os.path.relpath(pdf_path, BASE_DIR).replace('\\', '/')
        
        print(f"\n[{idx}/{len(all_pdfs)}] 📄 {filename}")
        print(f"    Path: {relative_path}")
        
        # Read PDF
        document_text = extract_text_from_pdf(pdf_path)
        if not document_text:
            print(f"    ❌ Could not extract text")
            total_docs_failed += 1
            continue
        
        print(f"    ✅ Extracted {len(document_text):,} characters")
        
        # Chunk document
        chunks = text_splitter.split_text(document_text)
        print(f"    ✅ Split into {len(chunks)} chunks")
        
        if len(chunks) == 0:
            print(f"    ⚠️  No text found after chunking")
            total_docs_failed += 1
            continue
        
        # Add chunks to database
        chunks_added_this_doc = 0
        chunks_skipped_this_doc = 0
        
        for i, chunk_text in enumerate(chunks):
            # Create unique ID
            chunk_id = f"{filename}_{i}"
            
            # Check if already exists
            try:
                existing = collection.get(ids=[chunk_id])
                if existing['ids']:
                    chunks_skipped_this_doc += 1
                    total_chunks_skipped += 1
                    continue
            except:
                pass
            
            # Add to database (ChromaDB auto-generates embeddings)
            try:
                collection.add(
                    documents=[chunk_text],
                    metadatas=[{"source": relative_path}],
                    ids=[chunk_id]
                )
                chunks_added_this_doc += 1
                total_chunks_added += 1
                
                # Progress indicator every 50 chunks
                if total_chunks_added % 50 == 0:
                    print(f"    📊 Progress: {total_chunks_added} chunks added so far...")
                
            except Exception as e:
                print(f"    ❌ Error adding chunk {i}: {e}")
        
        # Summary for this document
        if chunks_added_this_doc > 0:
            print(f"    ✅ Added {chunks_added_this_doc} new chunks")
        if chunks_skipped_this_doc > 0:
            print(f"    ⏭️  Skipped {chunks_skipped_this_doc} existing chunks")
        
        total_docs_processed += 1
    
    # Final summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("✅ BUILD COMPLETE!")
    print("="*80)
    print(f"\n📊 STATISTICS:")
    print(f"   Documents processed: {total_docs_processed}/{len(all_pdfs)}")
    print(f"   Documents failed: {total_docs_failed}")
    print(f"   Chunks added: {total_chunks_added:,}")
    print(f"   Chunks skipped: {total_chunks_skipped:,}")
    print(f"   Total chunks in database: {collection.count():,}")
    print(f"   Time taken: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    
    print(f"\n📍 DATABASE LOCATION:")
    print(f"   {DB_PATH}")
    
    print(f"\n🔧 EMBEDDING MODEL:")
    print(f"   ChromaDB built-in (all-MiniLM-L6-v2)")
    print(f"   Dimensions: 384")
    print(f"   Type: Sentence Transformer")
    
    print(f"\n✅ COMPATIBILITY:")
    print(f"   ✓ Online mode (Gemini API)")
    print(f"   ✓ Offline mode (Ollama)")
    print(f"   ✓ Offline mode (Mistral)")
    print(f"   ✓ No API key required for embeddings")
    
    print(f"\n📝 NEXT STEPS:")
    print(f"   1. Update app.py to use this database")
    print(f"   2. Change DB_PATH to: legal_db_universal")
    print(f"   3. Change collection name to: legal_brain_universal")
    print(f"   4. Restart your application")
    
    print("\n" + "="*80)
    print("🎉 Universal database ready for use!")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Build interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
