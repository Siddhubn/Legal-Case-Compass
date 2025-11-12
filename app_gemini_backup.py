import os
import fitz  # PyMuPDF
import chromadb
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
from dotenv import load_dotenv
import json
import re
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import requests
from llama_cpp import Llama

# --- 1. INITIALIZATION ---

# Version marker for debugging
APP_VERSION = "2.0-TwoStage"
print(f"=== Legal Case Compass {APP_VERSION} ===")

# Load API Key from .env file
load_dotenv()
try:
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
except KeyError:
    print("FATAL ERROR: GEMINI_API_KEY not found in .env file.")
    exit()

# Setup Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads' # We'll create this folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Secret key for session cookies. Set FLASK_SECRET_KEY in .env for production.
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev_secret_change_me')

# Users DB (for auth and saved analyses)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_DB_PATH = os.path.join(BASE_DIR, 'users.sqlite3')

def get_db_connection():
    conn = sqlite3.connect(USERS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_user_db():
    conn = get_db_connection()
    cur = conn.cursor()
    # users table (create minimal then migrate to full schema)
    cur.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    ''')
    # analyses table
    cur.execute('''
    CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        filename TEXT,
        story_summary TEXT,
        legal_summary TEXT,
        analysis_json TEXT,
        sources_json TEXT,
        recommended_json TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    ''')
    conn.commit()
    # --- Schema migration: add profile columns if missing ---
    cur.execute("PRAGMA table_info(users)")
    cols = [r[1] for r in cur.fetchall()]
    add_cols = [
        ("full_name", "TEXT"),
        ("phone", "TEXT"),
        ("address_current", "TEXT"),
        ("address_permanent", "TEXT"),
        ("dob", "TEXT"),
        ("email", "TEXT")
    ]
    for col, coltype in add_cols:
        if col not in cols:
            try:
                cur.execute(f"ALTER TABLE users ADD COLUMN {col} {coltype}")
            except Exception as e:
                print(f"Warning: could not add column {col}: {e}")
    # ensure analyses.recommended_json exists (safe)
    cur.execute("PRAGMA table_info(analyses)")
    anal_cols = [r[1] for r in cur.fetchall()]
    if 'recommended_json' not in anal_cols:
        try:
            cur.execute("ALTER TABLE analyses ADD COLUMN recommended_json TEXT")
        except Exception as e:
            print(f"Warning: could not add analyses.recommended_json: {e}")
    conn.commit()
    conn.close()

# Initialize user DB
init_user_db()

embedding_model = "models/text-embedding-004"

# Configure generation model with safety settings to prevent blocking
from google.generativeai.types import HarmCategory, HarmBlockThreshold
safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

generation_model = genai.GenerativeModel(
    "gemini-2.0-flash-exp",  # Using latest flash model
    safety_settings=safety_settings
)

# Setup Mistral-7B-Instruct model for offline use (as fallback)
MISTRAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "offline-access", "mistral-7b-instruct-v0.2.Q4_K_M.gguf")
llm_offline = None
if os.path.exists(MISTRAL_MODEL_PATH):
    try:
        llm_offline = Llama(
            model_path=MISTRAL_MODEL_PATH,
            n_gpu_layers=28,  # Increase for more GPU utilization (max 32 for Mistral)
            n_ctx=4096,
            verbose=False
        )
        print("[OK] Loaded Mistral-7B-Instruct model for offline use (fallback).")
    except Exception as e:
        print(f"[ERROR] Error loading Mistral model: {e}")
else:
    print(f"[WARNING] Offline model not found at {MISTRAL_MODEL_PATH}")

# Ollama Configuration (Primary offline option)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"  # Change to "neural-chat" or other models as needed
ollama_available = False

def check_ollama_availability():
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

# Check Ollama on startup
if check_ollama_availability():
    print("[OK] Ollama is available at localhost:11434")
    ollama_available = True
else:
    print("[WARNING] Ollama is not running. Will use Mistral model as fallback.")
    print("         To use Ollama, run: ollama serve")

# Manual switch for online/offline mode (None = auto-detect)
FORCE_MODE = 'offline'  # Set to 'online', 'offline', or None for auto
# NOTE: Set to 'offline' because Gemini API quota exceeded

# Connectivity check with manual override
def is_online():
    if FORCE_MODE == 'online':
        return True
    if FORCE_MODE == 'offline':
        return False
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except Exception:
        return False

# Setup Vector Database Connection
DB_PATH = os.path.join(BASE_DIR, "legal_db")

print("Connecting to Vector DB...")
try:
    db_client = chromadb.PersistentClient(path=DB_PATH)
    collection = db_client.get_collection("legal_brain")
    print("[OK] Connected to Vector DB.")
except Exception as e:
    print(f"[FATAL ERROR] Could not connect to ChromaDB at {DB_PATH}")
    print("Have you run the 'build_database.py' script first?")
    print(f"Error: {e}")
    exit()

# --- 2. HELPER FUNCTIONS ---

def extract_text_from_pdf(pdf_path):
    """Extracts all text from an uploaded PDF."""
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

def get_gemini_embedding(text):
    """Generates an embedding for a text chunk using Gemini."""
    try:
        return genai.embed_content(
            model=embedding_model,
            content=text,
            task_type="RETRIEVAL_QUERY" # Use QUERY type for searching
        )['embedding']
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def query_ollama(prompt, max_tokens=512):
    """Query Ollama API for text generation."""
    try:
        # Adjust timeout based on max_tokens
        timeout = 180 if max_tokens > 1500 else 120
        print(f"[DEBUG] Sending request to Ollama (timeout={timeout}s, max_tokens={max_tokens})...")
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "num_predict": max_tokens,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_thread": 8  # Use more CPU threads
                }
            },
            timeout=timeout
        )
        if response.status_code == 200:
            result = response.json().get("response", "")
            print(f"[OK] Ollama responded with {len(result)} chars")
            return result
        else:
            print(f"[ERROR] Ollama API error: {response.status_code}")
            print(f"[ERROR] Response: {response.text[:200]}")
            return None
    except requests.exceptions.Timeout:
        print(f"[ERROR] Ollama request timed out after 60 seconds")
        return None
    except Exception as e:
        print(f"[ERROR] Error querying Ollama: {e}")
        return None

def get_offline_response(prompt, max_tokens=512):
    """
    Get response from offline LLM.
    Tries Ollama first (GPU-accelerated), falls back to Mistral.
    """
    print(f"[DEBUG] Getting offline response (max_tokens={max_tokens})...")
    
    # Try Ollama first (preferred, GPU-accelerated) - check dynamically
    if check_ollama_availability():
        print("Using Ollama for offline inference...")
        try:
            response = query_ollama(prompt, max_tokens)
            if response:
                print(f"[OK] Ollama returned {len(response)} chars")
                return response
            else:
                print("[WARNING] Ollama returned empty response, falling back to Mistral...")
        except Exception as e:
            print(f"[ERROR] Ollama failed: {e}")
            print("Falling back to Mistral...")
    
    # Fallback to Mistral model
    if llm_offline:
        print("Using Mistral model for offline inference...")
        try:
            print("[DEBUG] Calling Mistral model...")
            response = llm_offline(prompt, max_tokens=max_tokens)
            result = response["choices"][0]["text"] if "choices" in response else response.get("text", "")
            print(f"[OK] Mistral returned {len(result)} chars")
            return result
        except Exception as e:
            print(f"[ERROR] Mistral model failed: {e}")
            return None
    
    print("[ERROR] No offline model available")
    return None

# --- 3. CORE APPLICATION LOGIC (THE "PIPELINE") ---

def get_summaries_from_text(document_text):
    """
    Uses Gemini if online, otherwise Ollama (or Mistral fallback) for summaries.
    """
    # Smart truncation: keep beginning and end, skip middle if too long
    max_chars = 20000  # ~4000 tokens (increased for better quality)
    if len(document_text) > max_chars:
        print(f"[INFO] Document is {len(document_text)} chars, using smart truncation...")
        # Keep first 60% and last 40% for context
        keep_start = int(max_chars * 0.6)
        keep_end = int(max_chars * 0.4)
        document_text = (
            document_text[:keep_start] + 
            "\n\n[... middle section omitted ...]\n\n" + 
            document_text[-keep_end:]
        )
        print(f"[INFO] Truncated to {len(document_text)} chars (kept beginning and end)")
    
    print("Generating summaries...")
    sys.stdout.flush()
    if is_online():
        print("[ONLINE] MODE: Using Gemini API")
        try:
            # Prompt 1: Story Summary
            prompt_story = f"""
            Read the following court document. Explain what happened in the case 
            in 1-2 paragraphs, using simple, non-legal language. 
            Describe the events like a straightforward narrative or story.

            DOCUMENT:
            {document_text}
            """
            story_response = generation_model.generate_content(prompt_story)

            # Prompt 2: Legal Summary
            prompt_legal = f"""
            Act as an expert legal analyst. Read the following court document and
            extract all key legal information. List all cited IPC/CPC sections,
            key dates, and a summary of the most recent actions or judgments.
            Be concise and formal. Output in one single paragraph.

            DOCUMENT:
            {document_text}
            """
            legal_response = generation_model.generate_content(prompt_legal)

            return story_response.text, legal_response.text
        except Exception as e:
            print(f"Error generating summaries: {e}")
            return "Error: Could not generate story summary.", "Error: Could not generate legal summary."
    else:
        print("[OFFLINE] MODE: Using Ollama/Mistral")
        # Use offline LLM (Ollama or Mistral)
        # Prompt 1: Story Summary - Make it very simple and narrative-like
        prompt_story = f"""
You are a storyteller explaining a court case to someone with no legal background.

Read this court document carefully and explain what happened in simple, everyday language.
Write 2-3 paragraphs that tell the story of what happened, who was involved, what they wanted, and what the court decided.

Use very simple words. Avoid legal jargon. If you must use a legal term, explain it in parentheses.
Write as if you're explaining to a friend or family member.

DOCUMENT:
{document_text}

Now tell the story in simple terms:
"""
        story_text = get_offline_response(prompt_story, max_tokens=600)
        if not story_text:
            story_text = "Error: Offline model not available or failed to generate response."

        # Prompt 2: Legal Summary - Focus on key facts and timelines
        prompt_legal = f"""
You are a legal analyst. Read this court case and provide a formal legal summary.

Include:
1. Who is involved (plaintiff/appellant vs defendant/respondent)
2. What laws were used (mention specific sections like Section 4, Section 6, etc.)
3. Important dates and timeline
4. What the courts decided at each level (High Court, Supreme Court, etc.)
5. The final ruling and what it means

Write as one detailed paragraph. Be specific about law sections and dates.

DOCUMENT:
{document_text}

Legal summary:
"""
        legal_text = get_offline_response(prompt_legal, max_tokens=600)
        if not legal_text:
            legal_text = "Error: Offline model not available or failed to generate response."

        return story_text.strip(), legal_text.strip()


def get_rag_answer(legal_summary, story_summary, user_role='unknown', original_doc_text=''):
    """
    Uses Gemini if online, otherwise Ollama (or Mistral fallback) for RAG.
    Now generates analysis in two stages for better quality.
    Takes legal_summary, story_summary, user_role, and original_doc_text as parameters.
    original_doc_text is used to create a deterministic hash for consistent source retrieval.
    """
    print(f"Generating RAG answer with two-stage generation (role: {user_role})...")
    sys.stdout.flush()
    
    # Determine perspective based on role
    role_context = ""
    if user_role == 'plaintiff':
        role_context = "The user is the PLAINTIFF/PETITIONER/APPELLANT. Focus on strategies and arguments that favor the plaintiff's position. Highlight precedents where plaintiffs succeeded."
    elif user_role == 'defendant':
        role_context = "The user is the DEFENDANT/RESPONDENT. Focus on defense strategies and arguments that favor the defendant's position. Highlight precedents where defendants succeeded."
    else:
        role_context = "Provide balanced analysis for both sides."
    
    print(f"[INFO] Analysis perspective: {role_context[:80]}...")
    sys.stdout.flush()

    # --- 3.A: Create Deterministic Search Query ---
    print("Creating deterministic search query for consistent results...")
    sys.stdout.flush()
    
    # Create a hash from ORIGINAL DOCUMENT for true consistency
    # This ensures same document = same hash = same sources ALWAYS
    import hashlib
    
    # Use original document if available, otherwise fall back to legal summary
    text_for_hash = original_doc_text if original_doc_text else legal_summary
    content_hash = hashlib.md5(text_for_hash.encode()).hexdigest()
    print(f"[DEBUG] Document content hash: {content_hash[:12]} (from {'original doc' if original_doc_text else 'legal summary'})")
    
    # Extract key elements from ORIGINAL DOCUMENT (not AI summary) for consistency
    # This ensures the search query is based on actual document content, not AI interpretation
    text_for_query = original_doc_text if original_doc_text else legal_summary
    
    # Extract key elements that define the case plot
    sections = sorted(set(re.findall(r'Section \d+[A-Z]*', text_for_query)))[:5]
    codes = sorted(set(re.findall(r'IPC|CPC|Cr\.P\.C\.|Navy Act|Army Act|Constitution', text_for_query)))[:3]
    
    # Extract plot elements (actions and outcomes)
    actions = sorted(set(re.findall(
        r'convicted|acquitted|dismissed|allowed|granted|denied|quashed|upheld|reversed|remanded|discharged',
        text_for_query, re.IGNORECASE
    )))[:3]
    
    subjects = sorted(set(re.findall(
        r'murder|robbery|theft|fraud|negligence|assault|corruption|sanction|bail|custody|appeal|prosecution',
        text_for_query, re.IGNORECASE
    )))[:3]
    
    # Build DETERMINISTIC query (same document = same query = same results ALWAYS)
    query_parts = []
    if sections:
        query_parts.extend(sections)
    if codes:
        query_parts.extend(codes)
    if subjects:
        query_parts.extend([s.lower() for s in subjects])
    if actions:
        query_parts.extend([a.lower() for a in actions])
    
    # Create deterministic query string
    deterministic_query = ' '.join(query_parts) if query_parts else text_for_query[:500]
    
    # Use DOCUMENT HASH as query hash for 100% consistency
    # Same document = same hash = same cached sources
    query_hash = content_hash[:12]
    
    print(f"[DEBUG] Query hash: {query_hash} (SAME document = SAME hash = SAME sources)")
    print(f"[DEBUG] Query: {deterministic_query[:100]}...")
    sys.stdout.flush()
    
    # Note: Not using embeddings for consistency - text search is more deterministic
    print("[INFO] Using text-based search for 100% consistency")
    sys.stdout.flush()

    # --- 3.B: Check Cache First (for 100% consistency) ---
    cache_file = os.path.join(BASE_DIR, "search_cache.json")
    cached_sources = None
    
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache = json.load(f)
                if query_hash in cache:
                    cached_sources = cache[query_hash]
                    print(f"[CACHE HIT] Found cached sources for query hash: {query_hash}")
                    print(f"[INFO] Using cached sources (ensures 100% consistency)")
                    sys.stdout.flush()
    except Exception as e:
        print(f"[WARNING] Could not read cache: {e}")
    
    # --- 3.C: Search the Vector Database (Deterministic Text Search) ---
    # Initialize sources and context
    sources = []
    context = ""
    
    if cached_sources:
        # Use cached sources
        sources = cached_sources
        print(f"[OK] Loaded {len(sources)} sources from cache")
        
        # Get documents for these sources
        try:
            # Query by source filenames
            all_docs = []
            for src in sources:
                results = collection.get(
                    where={"source": src},
                    limit=1
                )
                if results['documents']:
                    all_docs.append(results['documents'][0])
            
            if all_docs:
                context = "\n---\n".join([doc[:1500] for doc in all_docs])
            else:
                context = "Cached sources not found in database."
        except Exception as e:
            print(f"[WARNING] Could not retrieve cached documents: {e}")
            context = "Error retrieving cached documents."
    else:
        # Perform search
        print("Searching database for cases with similar plot and legal concepts...")
        sys.stdout.flush()
        
        try:
            print("[INFO] Using keyword-based search (compatible with Gemini embeddings)...")
            sys.stdout.flush()
            
            # Since database was built with Gemini embeddings (768 dim),
            # we need to use Gemini embeddings for search too
            # Create embedding from deterministic query
            try:
                print("[DEBUG] Creating Gemini embedding for search...")
                sys.stdout.flush()
                query_embedding = get_gemini_embedding(deterministic_query)
                
                if query_embedding:
                    print("[OK] Embedding created, searching...")
                    sys.stdout.flush()
                    
                    search_results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=20,
                        include=["documents", "metadatas", "distances"]
                    )
                else:
                    raise Exception("Could not create embedding")
                    
            except Exception as embed_error:
                print(f"[ERROR] Embedding failed: {embed_error}")
                print("[INFO] Cannot search without embeddings (database uses Gemini 768-dim)")
                sys.stdout.flush()
                
                # Set empty results and continue with analysis
                context = "Unable to search database (embedding dimension mismatch). Providing general analysis based on the case details."
                sources = []
                search_results = {'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
            
            # Process search results if we have any
            if search_results.get('documents') and search_results['documents'][0]:
                documents = search_results['documents'][0]
                metadatas = search_results['metadatas'][0]
                distances = search_results.get('distances', [[]])[0] if 'distances' in search_results else [0] * len(documents)
                
                print(f"[INFO] Analyzing {len(documents)} potential matches...")
                sys.stdout.flush()
                
                # Filter by similarity - Get TOP 7 most similar cases
                # IMPORTANT: Results are sorted by distance (lower = more similar)
                # This ensures we ALWAYS get the same top 7 cases for the same document
                relevant_docs = []
                relevant_sources = []
                seen = set()
                
                # Sort by distance (lower = more similar) - DETERMINISTIC ORDER
                sorted_results = sorted(zip(documents, metadatas, distances), key=lambda x: x[2])
                
                for doc, meta, dist in sorted_results:
                    src = meta['source']
                    
                    # Stop after exactly 7 sources (top 7 most similar)
                    if len(relevant_sources) >= 7:
                        break
                    
                    # Only include if:
                    # 1. Similar enough (distance < 1.1 for quality)
                    # 2. Not duplicate
                    # 3. Has meaningful content
                    if dist < 1.1 and src not in seen and len(doc.strip()) > 100:
                        # Categorize similarity
                        if dist < 0.6:
                            similarity = "EXCELLENT"
                        elif dist < 0.85:
                            similarity = "GOOD"
                        else:
                            similarity = "ACCEPTABLE"
                        
                        relevant_docs.append(doc[:1500] + "..." if len(doc) > 1500 else doc)
                        relevant_sources.append(src)
                        seen.add(src)
                        
                        filename = os.path.basename(src)
                        print(f"[{similarity}] {filename} (similarity: {1-dist:.1%}, distance: {dist:.3f})")
                        sys.stdout.flush()
                
                if relevant_docs:
                    context = "\n---\n".join(relevant_docs)
                    sources = relevant_sources
                    
                    # Cache the sources for future consistency
                    try:
                        cache = {}
                        if os.path.exists(cache_file):
                            with open(cache_file, 'r') as f:
                                cache = json.load(f)
                        
                        cache[query_hash] = sources
                        
                        with open(cache_file, 'w') as f:
                            json.dump(cache, f, indent=2)
                        
                        print(f"[CACHE] Saved {len(sources)} sources for query hash: {query_hash}")
                    except Exception as e:
                        print(f"[WARNING] Could not save cache: {e}")
                    
                    # Limit context length
                    max_context_length = 12000
                    if len(context) > max_context_length:
                        chars_per_doc = max_context_length // len(relevant_docs)
                        relevant_docs = [doc[:chars_per_doc] for doc in relevant_docs]
                        context = "\n---\n".join(relevant_docs)
                    
                    print(f"\n[SUCCESS] Found {len(sources)} similar cases (max 7 for quality)")
                    print(f"[INFO] Query hash: {query_hash} (cached for consistency)")
                    print(f"[INFO] Sources: {[os.path.basename(s) for s in sources]}")
                else:
                    print(f"[WARNING] No sufficiently similar cases found")
                    context = "No highly similar cases found. Providing analysis based on general legal principles."
                    sources = []
                
                sys.stdout.flush()
            else:
                # No documents found in search results
                print("[INFO] No documents returned from vector search")
                context = "No similar cases found in database. Providing analysis based on case details."
                sources = []
                
        except TimeoutError:
            print(f"[ERROR] Database search timed out")
            context = "Database timeout. Providing general analysis."
            sources = []
        except Exception as e:
            print(f"[ERROR] Database search failed: {e}")
            context = "Error searching database."
            sources = []


    # --- 3.C: Generate the Final Answer (as JSON) - STAGE 1: Basic Analysis ---
    print("Generating basic analysis with RAG...")
    
    # Log which mode will be used for RAG answer
    if is_online():
        print("[ONLINE] Using Gemini API for RAG analysis")
    else:
        print("[OFFLINE] Using Ollama/Mistral for RAG analysis")

    # STAGE 1: Generate basic analysis (what happened + key points)
    # Keep more of the legal summary for better quality
    legal_summary_truncated = legal_summary[:3000] if len(legal_summary) > 3000 else legal_summary
    
    basic_prompt = f"""You are a legal analyst. Analyze this court case and output ONLY valid JSON.

USER ROLE: {role_context}

CASE DETAILS:
{legal_summary_truncated}

SIMILAR PRECEDENTS FROM DATABASE:
{context[:3000]}

TASK: Create JSON with exactly 2 fields:

1. "what_has_happened": Write ONE paragraph (4-6 sentences) explaining:
   - What the court decided
   - The outcome for the {user_role}
   - What happens next
   - Be specific about dates, parties, court names
   - Reference similar cases if relevant

2. "key_legal_points": Write ONE paragraph (4-6 sentences) explaining:
   - Main IPC/CPC sections involved
   - What each section means in simple terms
   - How they apply to this case
   - How similar cases interpreted these sections
   - Implications for the {user_role}

CRITICAL RULES:
- Output ONLY the JSON object, nothing else
- Start with {{ and end with }}
- Use double quotes for strings
- Each field must be a single paragraph string (not an array)
- Reference the similar cases provided above
- Be specific and factual

EXAMPLE FORMAT:
{{"what_has_happened": "The Supreme Court ruled on April 26, 2016 in favor of the petitioner. The Court held that prosecution under Section 304-A IPC requires prior sanction under Section 197 Cr.P.C. Similar cases like Ram Kumar vs State (2014) established this principle. The case cannot proceed without government sanction.", "key_legal_points": "Section 197 Cr.P.C. requires government sanction before prosecuting public servants for official acts. Section 304-A IPC deals with causing death by negligence. The Supreme Court in Matajog Dobey (1956) held that Section 197 protection extends to negligent acts in official capacity. This means the prosecution must first obtain sanction from the competent authority."}}

Now generate the JSON for this case:
"""

    try:
        # STAGE 1: Generate basic analysis
        if is_online():
            basic_response = generation_model.generate_content(basic_prompt)
            basic_raw_text = basic_response.text
            # Log if response was blocked
            if hasattr(basic_response, 'prompt_feedback'):
                print(f"Prompt feedback: {basic_response.prompt_feedback}")
        else:
            basic_raw_text = get_offline_response(basic_prompt, max_tokens=1024)
            if not basic_raw_text:
                print("Offline model not available.")
                return {"error": "Offline model not available."}, []

        # Parse Stage 1 response - try multiple extraction methods
        print(f"Stage 1 raw response length: {len(basic_raw_text)} chars")
        
        # Method 1: Look for JSON between curly braces
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', basic_raw_text, re.DOTALL)
        if not match:
            # Method 2: Try to find JSON with nested structures
            match = re.search(r'\{.*\}', basic_raw_text, re.DOTALL)
        
        if not match:
            print(f"STAGE 1 JSON ERROR: No JSON found")
            print(f"Raw response (first 1000 chars): {basic_raw_text[:1000]}")
            # Try to create a basic response from the text
            analysis_data = {
                "what_has_happened": basic_raw_text[:500] if basic_raw_text else "Unable to parse response.",
                "key_legal_points": ["Unable to extract legal points from AI response."]
            }
        else:
            try:
                json_str = match.group(0)
                # Clean up common JSON issues
                json_str = json_str.replace('\n', ' ').replace('\r', '')
                analysis_data = json.loads(json_str)
                print("[OK] Stage 1 complete: what_has_happened + key_legal_points")
            except json.JSONDecodeError as je:
                print(f"STAGE 1 JSON PARSE ERROR: {je}")
                print(f"Attempted to parse: {json_str[:500]}")
                # Fallback
                analysis_data = {
                    "what_has_happened": "The AI response could not be parsed properly. Please try again.",
                    "key_legal_points": "JSON parsing failed. Please retry the analysis."
                }

        # STAGE 2: Generate detailed follow-ups based on Stage 1 + similar cases
        print("Generating detailed follow-ups based on similar cases...")
        
        # Truncate to avoid timeout
        context_short = context[:3000] if len(context) > 3000 else context
        case_short = analysis_data.get('what_has_happened', '')[:400]
        laws_short = analysis_data.get('key_legal_points', '')[:400]
        
        followup_prompt = f"""You are a legal strategist advising a {user_role}. Create strategic recommendations based on this case and similar precedents.

CURRENT CASE SUMMARY:
{case_short}

LEGAL ISSUES:
{laws_short}

SIMILAR PRECEDENTS FROM DATABASE:
{context_short}

TASK: Create JSON with "general_follow_ups" field containing an array of 6-8 detailed paragraphs.

Each paragraph should:
- Start with a clear topic (e.g., "Immediate action:", "Document gathering:", "Legal strategy:")
- Be 3-5 sentences long
- Reference specific similar cases from the database above
- Provide concrete, actionable advice
- Include timeframes where relevant
- Focus on strategies that worked in similar cases

TOPICS TO COVER:
1. Immediate actions needed (with timeline)
2. Documents to collect (based on what helped in similar cases)
3. Legal strategy (what worked in similar precedents)
4. Timeline management (court deadlines, filing requirements)
5. Evidence preparation (what was crucial in similar cases)
6. Procedural steps (based on similar case outcomes)
7. Common pitfalls to avoid (lessons from similar cases)
8. Rights protection (based on precedents)

CRITICAL RULES:
- Output ONLY the JSON object
- Start with {{ and end with }}
- "general_follow_ups" must be an array of strings
- Each string is one complete paragraph
- Reference specific similar cases by name
- Be specific and practical, not generic

EXAMPLE FORMAT:
{{"general_follow_ups": ["Immediate action: Based on the Supreme Court's ruling in your favor, act within 7 days. In the similar case of Ram Kumar vs State (2014), the petitioner's quick action prevented the prosecution from obtaining retrospective sanction. Engage a lawyer specializing in Section 197 cases immediately. Document all communications with your department.", "Document gathering: Collect all official records from your service period immediately. In Prakash Singh Badal vs State (2007), comprehensive documentation proved crucial. Request: (1) duty rosters for the relevant period, (2) written orders defining your responsibilities, (3) correspondence about vehicle allocation, (4) service records showing your official capacity. Similar cases show that documentary evidence is decisive.", "Legal strategy: The Supreme Court's finding that your act was in official capacity provides strong protection. However, monitor for attempts to obtain sanction retrospectively. In Matajog Dobey (1956), the Court held that Section 197 protection is absolute for official acts. File for permanent closure and seek costs from prosecution. Consider seeking expungement of records from your service file, as done successfully in similar cases."]}}

Now generate the JSON with 6-8 detailed paragraphs referencing the similar cases provided above:
"""

        if is_online():
            followup_response = generation_model.generate_content(followup_prompt)
            followup_raw_text = followup_response.text
            print(f"Stage 2 raw response length: {len(followup_raw_text)} chars")
            # Log if response was blocked
            if hasattr(followup_response, 'prompt_feedback'):
                print(f"Prompt feedback: {followup_response.prompt_feedback}")
        else:
            followup_raw_text = get_offline_response(followup_prompt, max_tokens=2048)
            if not followup_raw_text:
                print("Could not generate follow-ups")
                analysis_data["general_follow_ups"] = ["Unable to generate detailed follow-ups. Please consult a lawyer."]
                followup_raw_text = None

        # Parse Stage 2 response (both online and offline)
        if followup_raw_text:
            # Try multiple extraction methods
            match2 = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', followup_raw_text, re.DOTALL)
            if not match2:
                match2 = re.search(r'\{.*\}', followup_raw_text, re.DOTALL)
            
            if match2:
                try:
                    json_str2 = match2.group(0)
                    json_str2 = json_str2.replace('\n', ' ').replace('\r', '')
                    followup_data = json.loads(json_str2)
                    analysis_data["general_follow_ups"] = followup_data.get("general_follow_ups", [])
                    
                    if analysis_data["general_follow_ups"]:
                        print(f"[OK] Stage 2 complete: {len(analysis_data['general_follow_ups'])} follow-up items generated")
                    else:
                        print("[WARNING] Stage 2: Empty follow-ups array")
                        analysis_data["general_follow_ups"] = ["No specific follow-ups were generated. Please consult a qualified lawyer."]
                except json.JSONDecodeError as je2:
                    print(f"STAGE 2 JSON PARSE ERROR: {je2}")
                    print(f"Attempted to parse: {json_str2[:500]}")
                    analysis_data["general_follow_ups"] = ["Unable to parse follow-ups. Please try uploading the document again."]
            else:
                print("STAGE 2 JSON ERROR: No JSON found")
                print(f"Raw response (first 1000 chars): {followup_raw_text[:1000]}")
                analysis_data["general_follow_ups"] = ["Unable to extract follow-ups from AI response. Please try again."]

        # Ensure all fields exist
        if "general_follow_ups" not in analysis_data or not analysis_data["general_follow_ups"]:
            analysis_data["general_follow_ups"] = ["No specific follow-ups were generated. Please consult a qualified lawyer for guidance on next steps."]

        # Add disclaimer
        analysis_data["disclaimer"] = ("Disclaimer: This is not legal advice. I am an AI assistant. "
                                       "You must consult a qualified lawyer for advice on your specific case.")

        print(f"[OK] Analysis complete - Fields: {list(analysis_data.keys())}")
        print(f"[OK] Follow-ups count: {len(analysis_data.get('general_follow_ups', []))}")

        return analysis_data, sources

    except json.JSONDecodeError as e:
        print(f"JSON DECODE ERROR: {e}")
        return {
            "what_has_happened": "Error: The AI returned an invalid format. Please try uploading the document again.",
            "key_legal_points": "Unable to parse legal points due to formatting error.",
            "general_follow_ups": ["Unable to generate follow-ups due to formatting error."],
            "disclaimer": "An error occurred during analysis."
        }, []
    except Exception as e:
        print(f"Error generating analysis: {e}")
        return {
            "what_has_happened": f"Error: Could not generate the analysis. {e}",
            "key_legal_points": "Analysis generation failed.",
            "general_follow_ups": ["Please try again or contact support."],
            "disclaimer": "An error occurred."
        }, []


# --- 3.D: Simple category detector + sample lawyers ---
SAMPLE_LAWYERS = {
    'criminal': [
        {'name': 'A. Sharma', 'phone': '+91-90000-00001', 'won': '78%', 'ranking': 'City A', 'fees': '₹5,000'},
        {'name': 'R. Singh', 'phone': '+91-90000-00002', 'won': '65%', 'ranking': 'City B', 'fees': '₹4,000'},
    ],
    'family': [
        {'name': 'M. Gupta', 'phone': '+91-90000-00011', 'won': '70%', 'ranking': 'City A', 'fees': '₹3,500'},
        {'name': 'S. Rao', 'phone': '+91-90000-00012', 'won': '60%', 'ranking': 'City C', 'fees': '₹3,000'},
    ],
    'property': [
        {'name': 'D. Patel', 'phone': '+91-90000-00021', 'won': '82%', 'ranking': 'City B', 'fees': '₹6,000'},
        {'name': 'K. Verma', 'phone': '+91-90000-00022', 'won': '55%', 'ranking': 'City C', 'fees': '₹4,500'},
    ],
    'contract': [
        {'name': 'L. Menon', 'phone': '+91-90000-00031', 'won': '68%', 'ranking': 'City A', 'fees': '₹7,000'},
    ],
    'other': [
        {'name': 'P. Nair', 'phone': '+91-90000-00041', 'won': '50%', 'ranking': 'Regionwide', 'fees': '₹2,500'},
    ]
}


def detect_case_category(analysis_data, legal_summary_text):
    """Simple keyword-based detector returning one of the sample categories."""
    text = ''
    if isinstance(analysis_data, dict):
        # combine fields
        text += ' '.join([str(analysis_data.get(k, '')) for k in ['what_has_happened', 'key_legal_points', 'general_follow_ups'] if k in analysis_data])
    if legal_summary_text:
        text += ' ' + legal_summary_text
    text = text.lower()

    # keyword rules (simple)
    if any(k in text for k in ['murder', 'section 302', 'ipc', 'criminal', 'charge', 'arrest', 'bailable', 'non-bailable']):
        return 'criminal'
    if any(k in text for k in ['divorce', 'maintenance', 'custody', 'alimony', 'marriage', 'dowry']):
        return 'family'
    if any(k in text for k in ['property', 'land', 'title', 'real estate', 'mutation', 'possession']):
        return 'property'
    if any(k in text for k in ['contract', 'agreement', 'breach', 'commercial', 'service agreement']):
        return 'contract'
    return 'other'
    
# --- 4. FLASK WEB ROUTES ---

@app.route('/')
def home_page():
    """Serves the homepage/landing page."""
    return render_template('home.html')


@app.route('/app')
def app_page():
    """Serves the analysis UI page (formerly index)."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_document():
    """Handles the file upload and analysis."""
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    user_role = request.form.get('user_role', 'unknown')  # Get user's role
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not user_role or user_role == '':
        return jsonify({"error": "Please select your role in the case"}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        print(f"\n{'='*60}")
        print(f"--- New Job Started for {file.filename} ---")
        print(f"[INFO] User role: {user_role.upper()}")
        print(f"[DEBUG] File saved to: {filepath}")
        sys.stdout.flush()  # Force output immediately
        
        # --- Run the full pipeline ---
        
        # 1. Read PDF
        print("[STEP 1/3] Reading PDF...")
        sys.stdout.flush()
        doc_text = extract_text_from_pdf(filepath)
        if not doc_text:
            print("[ERROR] Could not extract text from PDF")
            sys.stdout.flush()
            return jsonify({"error": "Could not read text from PDF."}), 500
        print(f"[OK] Extracted {len(doc_text)} characters from PDF")
        sys.stdout.flush()
        
        # 2. Get Summaries
        print("[STEP 2/3] Generating summaries...")
        sys.stdout.flush()
        story_summary, legal_summary = get_summaries_from_text(doc_text)
        print(f"[OK] Story summary: {len(story_summary)} chars")
        print(f"[OK] Legal summary: {len(legal_summary)} chars")
        sys.stdout.flush()
        
        # 3. Get RAG Analysis (now with two-stage generation)
        print("[STEP 3/3] Generating RAG analysis...")
        sys.stdout.flush()
        analysis_data, sources = get_rag_answer(legal_summary, story_summary, user_role, doc_text)
        
        print("="*60)
        print("--- Job Complete ---")
        print(f"[RESULT] Analysis fields: {list(analysis_data.keys())}")
        print(f"[RESULT] Sources count: {len(sources)}")
        print(f"[RESULT] Final sources: {sources}")
        print("="*60)
        sys.stdout.flush()
        
        # 4. Auto-save analysis for logged-in users
        try:
            if session.get('user_id'):
                # detect category and recommendations
                category = detect_case_category(analysis_data, legal_summary)
                recommended = SAMPLE_LAWYERS.get(category, SAMPLE_LAWYERS['other'])

                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO analyses (user_id, filename, story_summary, legal_summary, analysis_json, sources_json, recommended_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        session.get('user_id'),
                        file.filename,
                        story_summary,
                        legal_summary,
                        json.dumps(analysis_data),
                        json.dumps(sources),
                        json.dumps({'category': category, 'recommended': recommended}),
                        datetime.utcnow().isoformat()
                    )
                )
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"Warning: could not save analysis to user DB: {e}")

        # 5. Return all results as JSON
        return jsonify({
            "storySummary": story_summary,
            "legalSummary": legal_summary,
            "analysis": analysis_data,  # <-- This is now an object, not a string
            "sources": sources,
            "recommended": {'category': detect_case_category(analysis_data, legal_summary), 'recommended': SAMPLE_LAWYERS.get(detect_case_category(analysis_data, legal_summary), SAMPLE_LAWYERS['other'])}
        })
    else:
        return jsonify({"error": "Invalid file type, please upload a PDF."}), 400

@app.route('/get-file/<path:filepath>')
def get_file(filepath):
    """
    Serves a file from the BASE_DIR.
    This is used to make the source links clickable.
    """
    print(f"File requested: {filepath}")
    try:
        # This securely serves a file from your project's base directory
        return send_from_directory(BASE_DIR, filepath, as_attachment=False)
    except Exception as e:
        print(f"Error serving file: {e}")
        return "File not found.", 404


@app.route('/register', methods=['POST'])
def register():
    """Registers a new user. Expects JSON with profile fields.
    Required: username, password
    Optional: full_name, phone, address_current, address_permanent, dob, email
    """
    # Accept JSON API or standard form POST
    data = request.get_json(silent=True)
    if not data:
        data = request.form

    username = (data.get('username') or '').strip()
    password = data.get('password') or ''
    full_name = (data.get('full_name') or '').strip()
    phone = (data.get('phone') or '').strip()
    address_current = (data.get('address_current') or '').strip()
    address_permanent = (data.get('address_permanent') or '').strip()
    dob = (data.get('dob') or '').strip()
    email = (data.get('email') or '').strip()

    if not username or not password:
        return jsonify({'error': 'Username and password are required.'}), 400

    pwd_hash = generate_password_hash(password)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''INSERT INTO users (username, password_hash, created_at, full_name, phone, address_current, address_permanent, dob, email)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (username, pwd_hash, datetime.utcnow().isoformat(), full_name, phone, address_current, address_permanent, dob, email))
        conn.commit()
        user_id = cur.lastrowid
        conn.close()
        # Log user in
        session['user_id'] = user_id
        session['username'] = username
        # If original request was a form submit, redirect to dashboard
        if not request.is_json:
            return redirect(url_for('dashboard_page'))
        return jsonify({'ok': True, 'username': username})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already exists.'}), 400
    except Exception as e:
        print(f"Error registering user: {e}")
        return jsonify({'error': 'Internal error.'}), 500


@app.route('/login', methods=['POST'])
def login():
    """Logs a user in. Expects JSON: {username, password}"""
    data = request.get_json(silent=True)
    if not data:
        data = request.form
    username = (data.get('username') or '').strip()
    password = data.get('password') or ''

    if not username or not password:
        return jsonify({'error': 'Username and password are required.'}), 400

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
        row = cur.fetchone()
        conn.close()
        if row and check_password_hash(row['password_hash'], password):
            session['user_id'] = row['id']
            session['username'] = username
            if not request.is_json:
                return redirect(url_for('dashboard_page'))
            return jsonify({'ok': True, 'username': username})
        else:
            return jsonify({'error': 'Invalid credentials.'}), 401
    except Exception as e:
        print(f"Error during login: {e}")
        return jsonify({'error': 'Internal error.'}), 500


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    return jsonify({'ok': True})


@app.route('/my_analyses', methods=['GET'])
def my_analyses():
    """Returns list of analyses for the logged-in user."""
    if not session.get('user_id'):
        return jsonify({'error': 'Not authenticated.'}), 401

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # user profile
        cur.execute('SELECT id, username, full_name, phone, address_current, address_permanent, dob, email, created_at FROM users WHERE id = ?', (session.get('user_id'),))
        user = cur.fetchone()
        # analyses list
        cur.execute('SELECT id, filename, created_at, substr(story_summary,1,250) as preview FROM analyses WHERE user_id = ? ORDER BY created_at DESC', (session.get('user_id'),))
        rows = cur.fetchall()
        conn.close()
        analyses = [dict(r) for r in rows]
        user_profile = dict(user) if user else {}
        return jsonify({'ok': True, 'analyses': analyses, 'user': user_profile})
    except Exception as e:
        print(f"Error fetching analyses: {e}")
        return jsonify({'error': 'Internal error.'}), 500


@app.route('/load_analysis/<int:analysis_id>', methods=['GET'])
def load_analysis(analysis_id):
    """Return a saved analysis payload for the logged-in user."""
    if not session.get('user_id'):
        return jsonify({'error': 'Not authenticated.'}), 401
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM analyses WHERE id = ? AND user_id = ?', (analysis_id, session.get('user_id')))
        row = cur.fetchone()
        conn.close()
        if not row:
            return jsonify({'error': 'Not found.'}), 404
        # Reconstruct response similar to /analyze
        analysis_obj = json.loads(row['analysis_json']) if row['analysis_json'] else {}
        sources = json.loads(row['sources_json']) if row['sources_json'] else []
        # sqlite3.Row does not support .get(); use index access
        recommended = json.loads(row['recommended_json']) if row['recommended_json'] else None
        if not recommended:
            # compute on the fly
            cat = detect_case_category(analysis_obj, row['legal_summary'])
            recommended = {'category': cat, 'recommended': SAMPLE_LAWYERS.get(cat, SAMPLE_LAWYERS['other'])}
        return jsonify({
            'storySummary': row['story_summary'],
            'legalSummary': row['legal_summary'],
            'analysis': analysis_obj,
            'sources': sources,
            'recommended': recommended
        })
    except Exception as e:
        print(f"Error loading analysis: {e}")
        return jsonify({'error': 'Internal error.'}), 500


@app.route('/profile', methods=['GET'])
def get_profile():
    if not session.get('user_id'):
        return jsonify({'error': 'Not authenticated.'}), 401
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT id, username, full_name, phone, address_current, address_permanent, dob, email, created_at FROM users WHERE id = ?', (session.get('user_id'),))
        row = cur.fetchone()
        conn.close()
        if not row:
            return jsonify({'error': 'Not found.'}), 404
        return jsonify({'ok': True, 'user': dict(row)})
    except Exception as e:
        print(f"Error getting profile: {e}")
        return jsonify({'error': 'Internal error.'}), 500


@app.route('/profile', methods=['POST'])
def update_profile():
    if not session.get('user_id'):
        return jsonify({'error': 'Not authenticated.'}), 401
    data = request.get_json(force=True)
    allowed = ['full_name', 'phone', 'address_current', 'address_permanent', 'dob', 'email']
    updates = {k: (data.get(k) or '').strip() for k in allowed}
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''UPDATE users SET full_name=?, phone=?, address_current=?, address_permanent=?, dob=?, email=? WHERE id = ?''',
                    (updates['full_name'], updates['phone'], updates['address_current'], updates['address_permanent'], updates['dob'], updates['email'], session.get('user_id')))
        conn.commit()
        conn.close()
        return jsonify({'ok': True, 'user': updates})
    except Exception as e:
        print(f"Error updating profile: {e}")
        return jsonify({'error': 'Internal error.'}), 500


# --- Serve auth and dashboard pages ---
@app.route('/login', methods=['GET'])
def login_page():
    # If already logged in, redirect to dashboard
    if session.get('user_id'):
        return redirect(url_for('dashboard_page'))
    return render_template('login.html')


@app.route('/register', methods=['GET'])
def register_page():
    # If already logged in, redirect to dashboard
    if session.get('user_id'):
        return redirect(url_for('dashboard_page'))
    return render_template('register.html')


@app.route('/dashboard', methods=['GET'])
def dashboard_page():
    if not session.get('user_id'):
        return render_template('login.html')
    return render_template('dashboard.html')

# --- 5. RUN THE APP ---
if __name__ == '__main__':
    # Fix Windows console encoding issues and disable buffering
    import sys
    
    # Disable output buffering for real-time logs
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    if sys.platform == 'win32':
        try:
            # Set console to UTF-8 mode
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
        except Exception:
            pass  # If it fails, continue anyway
    
    # Runs the web server
    print("[INFO] Starting Flask server with real-time logging...")
    app.run(debug=True, port=5000, use_reloader=False)