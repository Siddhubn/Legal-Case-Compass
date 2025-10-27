import os
import fitz  # PyMuPDF
import chromadb
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv
import json
import re

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

# Setup Gemini Models
embedding_model = "models/text-embedding-004"
generation_model = genai.GenerativeModel("gemini-2.5-flash") # Using 1.5 Pro

# Setup Vector Database Connection
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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

# --- 3. CORE APPLICATION LOGIC (THE "PIPELINE") ---

def get_summaries_from_text(document_text):
    """
    Uses Gemini to generate the two summaries.
    This is Part A of our plan.
    """
    print("Generating summaries...")
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


def get_rag_answer(legal_summary):
    """
    Uses the legal summary to search the DB and generate the final answer
    in a structured JSON format.
    """
    print("Generating RAG answer...")
    
    # --- 3.A: Embed the User's Query ---
    query_vector = get_gemini_embedding(legal_summary)
    if not query_vector:
        return {"error": "Could not create an embedding for your document."}, []

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
    
    # SLIGHTLY UPDATED PROMPT
    final_prompt = f"""
    You are a helpful legal AI assistant. You cannot give legal advice.
    Your goal is to provide information and general options based on the
    user's document and relevant legal context.

    **User's Case Summary:**
    {legal_summary}

    **Relevant Legal Information (from past cases, IPC, CPC, Constitution):**
    {context}

    **Your Task:**
    Based *only* on the User's Case Summary and the Relevant Legal Information
    provided above, generate a JSON object with three specific keys:
    1. "what_has_happened": A simple, plain-text explanation of the case status.
    2. "key_legal_points": A plain-text, bulleted list (using '*' or '-') of the main legal sections or principles.
    3. "general_follow_ups": A plain-text, bulleted list (using '*' or '-') of general, safe next steps.
    
    **CRITICAL:** ONLY output the raw JSON object. Your entire response must
    start with {{ and end with }}. Do not add *any* text before or after.
    """
    
    try:
        final_response = generation_model.generate_content(final_prompt)
        
        # --- NEW, MORE ROBUST JSON PARSING ---
        raw_text = final_response.text
        
        # Use regex to find the JSON block, even if the AI adds text
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        
        if not match:
            print(f"JSON DECODE ERROR: No JSON object found in response.")
            print(f"Raw response was: {raw_text}")
            raise json.JSONDecodeError("No JSON object found in AI response.", raw_text, 0)

        clean_json_string = match.group(0)
        analysis_data = json.loads(clean_json_string)
        # --- END OF NEW PARSING ---
        
        # Add the disclaimer to the data
        analysis_data["disclaimer"] = ("Disclaimer: This is not legal advice. I am an AI assistant. "
                                       "You must consult a qualified lawyer for advice on your specific case.")
        
        return analysis_data, sources

    except json.JSONDecodeError as e:
        # This will now catch both "No JSON" and "Malformed JSON"
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
    
# --- 4. FLASK WEB ROUTES ---

@app.route('/')
def index():
    """Serves the main HTML page."""
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
        
        # 4. Return all results as JSON
        return jsonify({
            "storySummary": story_summary,
            "legalSummary": legal_summary,
            "analysis": analysis_data,  # <-- This is now an object, not a string
            "sources": sources
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

# --- 5. RUN THE APP ---
if __name__ == '__main__':
    # Runs the web server
    app.run(debug=True, port=5000)