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
generation_model = genai.GenerativeModel("gemini-2.5-flash") # Using 1.5 Pro

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
        print("✅ Loaded Mistral-7B-Instruct model for offline use (fallback).")
    except Exception as e:
        print(f"Error loading Mistral model: {e}")
else:
    print(f"Offline model not found at {MISTRAL_MODEL_PATH}")

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
    print("✅ Ollama is available at localhost:11434")
    ollama_available = True
else:
    print("⚠️  Ollama is not running. Will use Mistral model as fallback.")
    print("   To use Ollama, run: ollama serve")

# Manual switch for online/offline mode (None = auto-detect)
FORCE_MODE = 'offline'  # Set to 'online', 'offline', or None for auto

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
    print("✅ Connected to Vector DB.")
except Exception as e:
    print(f"FATAL ERROR: Could not connect to ChromaDB at {DB_PATH}")
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
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "num_predict": max_tokens
            },
            timeout=300  # 5 minutes timeout for long responses
        )
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            print(f"Ollama API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return None

def get_offline_response(prompt, max_tokens=512):
    """
    Get response from offline LLM.
    Tries Ollama first (GPU-accelerated), falls back to Mistral.
    """
    # Try Ollama first (preferred, GPU-accelerated) - check dynamically
    if check_ollama_availability():
        print("Using Ollama for offline inference...")
        response = query_ollama(prompt, max_tokens)
        if response:
            return response
        else:
            print("Ollama query failed, falling back to Mistral...")
    
    # Fallback to Mistral model
    if llm_offline:
        print("Using Mistral model for offline inference...")
        try:
            response = llm_offline(prompt, max_tokens=max_tokens)
            return response["choices"][0]["text"] if "choices" in response else response["text"]
        except Exception as e:
            print(f"Error with Mistral model: {e}")
            return None
    
    return None

# --- 3. CORE APPLICATION LOGIC (THE "PIPELINE") ---

def get_summaries_from_text(document_text):
    """
    Uses Gemini if online, otherwise Ollama (or Mistral fallback) for summaries.
    """
    print("Generating summaries...")
    if is_online():
        print("📡 MODE: ONLINE (Using Gemini API)")
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
        print("🖥️  MODE: OFFLINE (Using Ollama/Mistral)")
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

Write as one paragraph. Be specific about law sections and dates.

DOCUMENT:
{document_text}

Legal summary:
"""
        legal_text = get_offline_response(prompt_legal, max_tokens=600)
        if not legal_text:
            legal_text = "Error: Offline model not available or failed to generate response."

        return story_text.strip(), legal_text.strip()


def get_rag_answer(legal_summary):
    """
    Uses Gemini if online, otherwise Ollama (or Mistral fallback) for RAG.
    """
    print("Generating RAG answer...")

    # --- 3.A: Embed the User's Query ---
    print("Creating embedding for case summary...")
    if is_online():
        print("📡 MODE: ONLINE (Using Gemini embeddings)")
        query_vector = get_gemini_embedding(legal_summary)
        if not query_vector:
            return {"error": "Could not create an embedding for your document."}, []
    else:
        # For offline, use the same embedding model as DB build (if available)
        # If DB was built with Gemini embeddings, offline retrieval may be less accurate
        print("🖥️  MODE: OFFLINE (Using Gemini embeddings for retrieval)")
        query_vector = get_gemini_embedding(legal_summary)
        if not query_vector:
            return {"error": "Could not create an embedding for your document (offline)."}, []

    # --- 3.B: Search the Vector Database ---
    print("Searching database for relevant cases...")
    try:
        search_results = collection.query(
            query_embeddings=[query_vector],
            n_results=5,  # Get the top 5 most similar chunks
            include=["documents", "metadatas"] # ASK FOR METADATA
        )
    except Exception as e:
        print(f"Error querying database: {e}")
        return {"error": f"Error querying database: {e}"}, []

    context = "\n---\n".join(search_results['documents'][0])
    metadatas = search_results['metadatas'][0]
    sources = list(dict.fromkeys([meta['source'] for meta in metadatas]))

    # --- 3.C: Generate the Final Answer (as JSON) ---
    print("Generating final answer with RAG...")
    
    # Log which mode will be used for RAG answer
    if is_online():
        print("📡 MODE: ONLINE (Using Gemini API for RAG analysis)")
    else:
        print("🖥️  MODE: OFFLINE (Using Ollama/Mistral for RAG analysis)")

    final_prompt = f"""
You are a helpful legal information assistant. Your job is to help someone understand their court case.

USER'S CASE SUMMARY:
{legal_summary}

SIMILAR CASES AND LEGAL INFORMATION FROM DATABASE:
{context}

YOUR TASK:
Analyze the user's case based on the summary and similar cases above. Then create a JSON response with three sections:

1. "what_has_happened": Write a clear explanation of what has happened in the case so far. Explain:
   - What the court has decided
   - What happens next (any actions needed)
   - Important deadlines or timelines
   Write in simple language, as if explaining to someone who is not a lawyer.

2. "key_legal_points": List the most important laws and legal concepts involved. For each point:
   - State the law section (like "Section 24 of the 2013 Act")
   - Explain briefly what it means in simple terms
   Use bullets (- or *). Keep each point short and clear.

3. "general_follow_ups": List important next steps the person should take:
   - Monitor important deadlines
   - Gather necessary documents
   - Actions recommended
   Write as a bulleted list. Focus on practical steps.

IMPORTANT: 
- Write everything in simple, everyday language
- Avoid legal jargon unless absolutely necessary
- If you use legal terms, explain them simply
- Be specific about dates, amounts, and deadlines if mentioned
- Output ONLY the JSON object with no text before or after
- Start with {{ and end with }}

Generate the JSON now:
"""

    try:
        if is_online():
            final_response = generation_model.generate_content(final_prompt)
            raw_text = final_response.text
        else:
            # Use offline LLM (Ollama or Mistral)
            raw_text = get_offline_response(final_prompt, max_tokens=1024)
            if not raw_text:
                print("Offline model not available.")
                return {"error": "Offline model not available."}, []

        # Use regex to find the JSON block, even if the AI adds text
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)

        if not match:
            print(f"JSON DECODE ERROR: No JSON object found in response.")
            print(f"Raw response was: {raw_text}")
            raise json.JSONDecodeError("No JSON object found in AI response.", raw_text, 0)

        clean_json_string = match.group(0)
        analysis_data = json.loads(clean_json_string)

        # Add the disclaimer to the data
        analysis_data["disclaimer"] = ("Disclaimer: This is not legal advice. I am an AI assistant. "
                                       "You must consult a qualified lawyer for advice on your specific case.")

        return analysis_data, sources

    except json.JSONDecodeError as e:
        print(f"JSON DECODE ERROR: {e}")
        return {
            "what_has_happened": "Error: The AI returned an invalid analysis format.",
            "key_legal_points": "Please try uploading the document again.",
            "general_follow_ups": "",
            "disclaimer": "An error occurred."
        }, []
    except Exception as e:
        print(f"Error generating final answer: {e}")
        return {
            "what_has_happened": f"Error: Could not generate the final analysis. {e}",
            "key_legal_points": "",
            "general_follow_ups": "",
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
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and file.filename.lower().endswith('.pdf'):
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        print(f"--- New Job Started for {file.filename} ---")
        
        # --- Run the full pipeline ---
        
        # 1. Read PDF
        doc_text = extract_text_from_pdf(filepath)
        if not doc_text:
            return jsonify({"error": "Could not read text from PDF."}), 500
        
        # 2. Get Summaries
        story_summary, legal_summary = get_summaries_from_text(doc_text)
        
        # 3. Get RAG Analysis
        analysis_data, sources = get_rag_answer(legal_summary) # <-- Renamed for clarity
        
        print("--- Job Complete ---")
        
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
    # Runs the web server
    app.run(debug=True, port=5000)