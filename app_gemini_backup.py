import os
import fitz  # PyMuPDF
import chromadb
import google.generativeai as genai
from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for, abort
from dotenv import load_dotenv
import json
import re
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import requests
from llama_cpp import Llama
import time  # For retry delays
import sys  # For stdout flushing
import secrets  # For secure token generation
import hashlib  # For file integrity checks
from functools import wraps  # For decorators
import bleach  # For XSS protection (install: pip install bleach)

# --- 1. INITIALIZATION ---

# Version marker for debugging
APP_VERSION = "2.0-TwoStage-SecureOWASP2025"
print(f"=== Legal Case Compass {APP_VERSION} ===")

# Define BASE_DIR early for logging
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load API Key from .env file
load_dotenv()
try:
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
except KeyError:
    print("FATAL ERROR: GEMINI_API_KEY not found in .env file.")
    exit()

# --- OWASP SECURITY: COMPREHENSIVE LOGGING (A09:2025 - Security Logging) ---
import logging
from logging.handlers import RotatingFileHandler

# Create logs directory if it doesn't exist
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure security logger
security_logger = logging.getLogger('security')
security_logger.setLevel(logging.INFO)
security_handler = RotatingFileHandler(
    os.path.join(LOGS_DIR, 'security.log'),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
security_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
security_handler.setFormatter(security_formatter)
security_logger.addHandler(security_handler)

# Configure application logger
app_logger = logging.getLogger('application')
app_logger.setLevel(logging.INFO)
app_handler = RotatingFileHandler(
    os.path.join(LOGS_DIR, 'app.log'),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
app_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
app_handler.setFormatter(app_formatter)
app_logger.addHandler(app_handler)

def log_security_event(event_type, details, ip_address=None, user_id=None, severity='INFO'):
    """Log security events for monitoring (A09:2025 - Security Logging)"""
    try:
        ip = ip_address or get_client_ip() if request else 'unknown'
        user = user_id or session.get('user_id', 'anonymous') if session else 'anonymous'
        message = f"[{event_type}] IP:{ip} User:{user} - {details}"
        
        if severity == 'WARNING':
            security_logger.warning(message)
        elif severity == 'ERROR':
            security_logger.error(message)
        elif severity == 'CRITICAL':
            security_logger.critical(message)
        else:
            security_logger.info(message)
    except Exception as e:
        print(f"[ERROR] Failed to log security event: {e}")

def log_app_event(message, level='INFO'):
    """Log application events"""
    try:
        if level == 'WARNING':
            app_logger.warning(message)
        elif level == 'ERROR':
            app_logger.error(message)
        elif level == 'CRITICAL':
            app_logger.critical(message)
        else:
            app_logger.info(message)
    except Exception as e:
        print(f"[ERROR] Failed to log application event: {e}")

# Setup Flask App
app = Flask(__name__)

# --- OWASP TOP 10 2025 SECURITY CONFIGURATIONS ---

# 1. Secure Session Configuration (A01:2025 - Broken Access Control)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_COOKIE_SECURE'] = True  # HTTPS only
app.config['SESSION_COOKIE_HTTPONLY'] = True  # No JavaScript access
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)  # Session timeout

# 2. File Upload Security (A03:2025 - Injection)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}  # Only PDF files
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 3. Security Headers (A05:2025 - Security Misconfiguration)
@app.after_request
def set_security_headers(response):
    """Add comprehensive security headers to all responses (OWASP A05:2025)"""
    # Prevent MIME type sniffing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    
    # Prevent clickjacking
    response.headers['X-Frame-Options'] = 'DENY'
    
    # Enable XSS protection (legacy browsers)
    response.headers['X-XSS-Protection'] = '1; mode=block'
    
    # Force HTTPS (HSTS)
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'
    
    # Content Security Policy - Strict but functional
    csp_directives = [
        "default-src 'self'",
        "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://cdn.jsdelivr.net",
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com",
        "font-src 'self' https://cdn.jsdelivr.net https://fonts.gstatic.com",
        "img-src 'self' data: https:",
        "connect-src 'self'",
        "frame-ancestors 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "upgrade-insecure-requests"
    ]
    response.headers['Content-Security-Policy'] = "; ".join(csp_directives)
    
    # Referrer policy
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    
    # Permissions policy (restrict dangerous features)
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=(), payment=(), usb=(), magnetometer=(), gyroscope=()'
    
    # Prevent caching of sensitive data
    if request.path.startswith('/api/') or request.path in ['/login', '/register', '/profile']:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    
    return response

# 4. Rate Limiting Storage (A07:2025 - Identification and Authentication Failures)
login_attempts = {}  # IP -> (attempts, last_attempt_time)
MAX_LOGIN_ATTEMPTS = 5
LOGIN_TIMEOUT = timedelta(minutes=15)

# Users DB (for auth and saved analyses)
# BASE_DIR already defined at top of file
USERS_DB_PATH = os.path.join(BASE_DIR, 'users.sqlite3')

# --- OWASP SECURITY HELPER FUNCTIONS ---

def allowed_file(filename):
    """Check if file extension is allowed (A03:2025 - Injection)"""
    if not filename or '.' not in filename:
        print(f"[DEBUG] allowed_file: No filename or no extension - {filename}")
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    allowed = ext in app.config['ALLOWED_EXTENSIONS']
    print(f"[DEBUG] allowed_file: filename={filename}, ext={ext}, allowed={allowed}, allowed_exts={app.config['ALLOWED_EXTENSIONS']}")
    return allowed

def sanitize_input(text, max_length=10000):
    """Sanitize user input to prevent XSS (A03:2025 - Injection)"""
    if not text:
        return text
    # Truncate to max length to prevent DoS
    text = str(text)[:max_length]
    # Remove HTML tags and dangerous characters
    cleaned = bleach.clean(text, tags=[], strip=True)
    # Additional sanitization: remove null bytes and control characters
    cleaned = cleaned.replace('\x00', '').replace('\r', '')
    return cleaned

def validate_filename(filename):
    """Validate and secure filename (A03:2025 - Injection)"""
    if not filename:
        print(f"[DEBUG] validate_filename: No filename provided")
        return None
    # Use werkzeug's secure_filename
    secured = secure_filename(filename)
    if not secured:
        print(f"[DEBUG] validate_filename: secure_filename returned empty for {filename}")
        return None
    # Additional validation: only alphanumeric, dots, dashes, underscores, spaces
    if not re.match(r'^[\w\-. ]+$', secured):
        print(f"[DEBUG] validate_filename: Regex failed for {secured}")
        return None
    # Prevent double extensions and hidden files
    if secured.startswith('.') or '..' in secured:
        print(f"[DEBUG] validate_filename: Hidden file or double extension: {secured}")
        return None
    # Limit filename length
    if len(secured) > 255:
        print(f"[DEBUG] validate_filename: Filename too long: {len(secured)} chars")
        return None
    print(f"[DEBUG] validate_filename: Valid filename: {secured}")
    return secured

def validate_email(email):
    """Validate email format (A03:2025 - Injection)"""
    if not email:
        return False
    # RFC 5322 compliant email regex (simplified)
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return False
    # Additional checks
    if len(email) > 254:  # RFC 5321
        return False
    local, domain = email.rsplit('@', 1)
    if len(local) > 64:  # RFC 5321
        return False
    return True

def validate_phone(phone):
    """Validate phone number format (A03:2025 - Injection)"""
    if not phone:
        return True  # Phone is optional
    # Allow international format with +, digits, spaces, dashes, parentheses
    pattern = r'^[+]?[0-9\s\-()]{10,20}$'
    return bool(re.match(pattern, phone))

def validate_username(username):
    """Validate username format (A03:2025 - Injection)"""
    if not username:
        return False
    # Only alphanumeric and underscore, 3-50 characters
    if not re.match(r'^[a-zA-Z0-9_]{3,50}$', username):
        return False
    # Prevent SQL keywords as usernames
    sql_keywords = ['select', 'insert', 'update', 'delete', 'drop', 'create', 'alter', 'exec', 'execute']
    if username.lower() in sql_keywords:
        return False
    return True

def validate_password_strength(password):
    """Validate password strength (A07:2025 - Auth Failures)"""
    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if len(password) > 128:
        return False, "Password must be less than 128 characters"
    # Check for at least one uppercase, one lowercase, one digit
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one digit"
    # Check for common weak passwords
    weak_passwords = ['password', '12345678', 'qwerty123', 'admin123', 'letmein1']
    if password.lower() in weak_passwords:
        return False, "Password is too common, please choose a stronger password"
    return True, "Password is strong"

def validate_integer_id(value, min_val=1, max_val=2147483647):
    """Validate integer ID (A03:2025 - Injection)"""
    try:
        int_val = int(value)
        return min_val <= int_val <= max_val
    except (ValueError, TypeError):
        return False

def check_rate_limit(ip_address):
    """Check if IP has exceeded login attempts (A07:2025 - Auth Failures)"""
    if not ip_address:
        return False
    
    # Clean up old entries periodically
    current_time = datetime.now()
    expired_ips = [ip for ip, (_, last_attempt) in login_attempts.items() 
                   if current_time - last_attempt > LOGIN_TIMEOUT]
    for ip in expired_ips:
        del login_attempts[ip]
    
    if ip_address in login_attempts:
        attempts, last_attempt = login_attempts[ip_address]
        # Reset if timeout has passed
        if current_time - last_attempt > LOGIN_TIMEOUT:
            del login_attempts[ip_address]
            return True
        # Check if max attempts exceeded
        if attempts >= MAX_LOGIN_ATTEMPTS:
            return False
    return True

def record_login_attempt(ip_address, success=False):
    """Record login attempt for rate limiting (A07:2025 - Auth Failures)"""
    if not ip_address:
        return
    
    if success:
        # Clear attempts on successful login
        if ip_address in login_attempts:
            del login_attempts[ip_address]
    else:
        # Increment failed attempts
        current_time = datetime.now()
        if ip_address in login_attempts:
            attempts, _ = login_attempts[ip_address]
            login_attempts[ip_address] = (attempts + 1, current_time)
        else:
            login_attempts[ip_address] = (1, current_time)

def get_client_ip():
    """Get client IP address safely (A01:2025 - Broken Access Control)"""
    # Check for proxy headers (in order of trust)
    if request.headers.get('X-Real-IP'):
        return request.headers.get('X-Real-IP')
    if request.headers.get('X-Forwarded-For'):
        # Take the first IP in the chain
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    return request.remote_addr or 'unknown'

def require_auth(f):
    """Decorator to require authentication (A01:2025 - Broken Access Control)"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if request.is_json or request.headers.get('Accept') == 'application/json':
                return jsonify({'error': 'Authentication required'}), 401
            return redirect(url_for('login_page'))
        
        # Validate session integrity
        if not validate_session():
            session.clear()
            if request.is_json or request.headers.get('Accept') == 'application/json':
                return jsonify({'error': 'Session expired or invalid'}), 401
            return redirect(url_for('login_page'))
        
        return f(*args, **kwargs)
    return decorated_function

def validate_session():
    """Validate session integrity (A07:2025 - Auth Failures)"""
    if 'user_id' not in session:
        return False
    
    # Check if session has required fields
    if not isinstance(session.get('user_id'), int):
        return False
    
    # Validate session hasn't been tampered with
    if 'username' not in session:
        return False
    
    return True

def validate_user_access(user_id, resource_user_id):
    """Validate user can access resource (A01:2025 - Broken Access Control)"""
    try:
        return int(user_id) == int(resource_user_id)
    except (ValueError, TypeError):
        return False

def validate_csrf_token():
    """Validate CSRF token for state-changing operations (A01:2025 - Broken Access Control)"""
    # For JSON API requests, check Origin/Referer headers
    if request.is_json:
        origin = request.headers.get('Origin')
        referer = request.headers.get('Referer')
        host = request.headers.get('Host')
        
        # Allow same-origin requests
        if origin and host:
            if not origin.endswith(f'//{host}'):
                return False
        elif referer and host:
            if not referer.startswith(f'http://{host}') and not referer.startswith(f'https://{host}'):
                return False
    
    return True

def get_db_connection():
    """Get database connection with security settings (A03:2025 - Injection Prevention)"""
    try:
        # SECURITY: Use absolute path to prevent path traversal
        db_path = os.path.abspath(USERS_DB_PATH)
        
        # SECURITY: Verify database file is in expected location
        if not db_path.startswith(os.path.abspath(BASE_DIR)):
            raise ValueError("Database path is outside base directory")
        
        conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10.0)
        conn.row_factory = sqlite3.Row
        
        # SECURITY: Enable foreign keys for referential integrity (A08:2025 - Data Integrity)
        conn.execute("PRAGMA foreign_keys = ON")
        
        # SECURITY: Enable secure delete to overwrite deleted data
        conn.execute("PRAGMA secure_delete = ON")
        
        # SECURITY: Set journal mode for better concurrency and crash recovery
        conn.execute("PRAGMA journal_mode = WAL")
        
        # SECURITY: Limit query execution time to prevent DoS
        conn.execute("PRAGMA busy_timeout = 5000")
        
        return conn
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        raise

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
    "gemini-2.5-flash",  # Using latest flash model
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
FORCE_MODE = None  # Set to 'online', 'offline', or None for auto
# NOTE: Set to 'offline' because Gemini API quota exceeded

# Connectivity check with manual override
def is_online(force_check=False):
    """
    Check if online mode is available (Gemini API accessible).
    
    Args:
        force_check: If True, always check connectivity even if cached
    
    Returns:
        bool: True if online mode available, False otherwise
    """
    # Respect manual override
    if FORCE_MODE == 'online':
        return True
    if FORCE_MODE == 'offline':
        return False
    
    # Use cached result if available and not forcing check
    if not force_check and hasattr(is_online, '_cached_result'):
        cache_time = getattr(is_online, '_cache_time', 0)
        # Cache is valid for 30 seconds
        if time.time() - cache_time < 30:
            return is_online._cached_result
    
    # Check connectivity to Gemini API
    try:
        # Quick check to Google (Gemini API host)
        response = requests.get("https://generativelanguage.googleapis.com", timeout=3)
        result = response.status_code in [200, 403, 404]  # Any response means online
        
        # Cache the result
        is_online._cached_result = result
        is_online._cache_time = time.time()
        
        if result:
            print("[CONNECTIVITY] Online mode available (Gemini API accessible)")
        else:
            print("[CONNECTIVITY] Online mode unavailable, will use offline mode")
        
        return result
    except Exception as e:
        print(f"[CONNECTIVITY] Cannot reach Gemini API: {str(e)[:50]}")
        print("[CONNECTIVITY] Switching to offline mode")
        
        # Cache the offline result
        is_online._cached_result = False
        is_online._cache_time = time.time()
        
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

def call_gemini_with_retry(model, prompt, max_retries=3, initial_wait=30, allow_fallback=True):
    """
    Call Gemini API with automatic retry on rate limit errors and fallback to offline mode.
    
    Args:
        model: Gemini model instance
        prompt: Prompt to send
        max_retries: Maximum number of retry attempts (default: 3)
        initial_wait: Initial wait time in seconds (default: 30)
        allow_fallback: If True, fallback to offline mode on failure (default: True)
    
    Returns:
        Response from Gemini API or None if failed and fallback not allowed
    """
    for attempt in range(max_retries):
        try:
            # Check connectivity before attempting
            if attempt > 0:
                if not is_online(force_check=True):
                    print(f"[CONNECTIVITY] Lost connection to Gemini API")
                    if allow_fallback:
                        print(f"[FALLBACK] Switching to offline mode")
                        return None  # Signal to use offline mode
                    raise Exception("Gemini API not accessible")
            
            response = model.generate_content(prompt)
            return response  # Success!
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a connectivity error
            if any(err in error_msg.lower() for err in ['connection', 'timeout', 'network', 'unreachable']):
                print(f"[CONNECTIVITY] Network error: {error_msg[:80]}")
                if allow_fallback:
                    print(f"[FALLBACK] Switching to offline mode due to connectivity issue")
                    # Update cached connectivity status
                    is_online._cached_result = False
                    is_online._cache_time = time.time()
                    return None  # Signal to use offline mode
                raise e
            
            # Check if it's a rate limit error
            if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                wait_time = initial_wait * (attempt + 1)  # Exponential backoff
                
                if attempt < max_retries - 1:  # Not the last attempt
                    print(f"[RATE LIMIT] Gemini quota exceeded (attempt {attempt + 1}/{max_retries})")
                    print(f"[WAITING] Waiting {wait_time} seconds for quota to reset...")
                    sys.stdout.flush()
                    
                    # Wait with progress indicator
                    for remaining in range(wait_time, 0, -10):
                        print(f"[WAITING] {remaining} seconds remaining...")
                        sys.stdout.flush()
                        time.sleep(min(10, remaining))
                    
                    print(f"[RETRY] Retrying Gemini API (attempt {attempt + 2}/{max_retries})...")
                    sys.stdout.flush()
                    continue  # Retry
                else:
                    # Last attempt failed
                    print(f"[ERROR] Gemini API failed after {max_retries} attempts")
                    if allow_fallback:
                        print(f"[FALLBACK] Switching to offline mode due to rate limit")
                        return None  # Signal to use offline mode
                    raise e
            else:
                # Other error
                print(f"[ERROR] Gemini API error: {error_msg[:100]}")
                if attempt < max_retries - 1:
                    print(f"[RETRY] Retrying (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(5)  # Short wait before retry
                    continue
                else:
                    if allow_fallback:
                        print(f"[FALLBACK] Switching to offline mode due to API error")
                        return None  # Signal to use offline mode
                    raise e
    
    # Should never reach here, but just in case
    if allow_fallback:
        print(f"[FALLBACK] Max retries exceeded, switching to offline mode")
        return None
    raise Exception("Max retries exceeded")

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

def check_document_legitimacy(document_text):
    """
    Checks if the uploaded document is a legitimate legal/court document.
    Returns: (is_legitimate: bool, message: str, confidence: str)
    """
    print("[LEGITIMACY CHECK] Validating document type...")
    sys.stdout.flush()
    
    # Quick keyword-based check first (fast, no AI needed)
    legal_keywords = [
        'court', 'judge', 'petition', 'appellant', 'respondent', 'plaintiff', 'defendant',
        'supreme court', 'high court', 'district court', 'tribunal', 'magistrate',
        'section', 'ipc', 'cpc', 'cr.p.c', 'constitution', 'act', 'law',
        'judgment', 'order', 'decree', 'writ', 'appeal', 'case', 'suit',
        'prosecution', 'accused', 'conviction', 'acquittal', 'bail', 'custody',
        'hearing', 'evidence', 'witness', 'counsel', 'advocate', 'lawyer'
    ]
    
    text_lower = document_text.lower()
    keyword_matches = sum(1 for keyword in legal_keywords if keyword in text_lower)
    
    # If very few legal keywords, likely not a legal document
    if keyword_matches < 3:
        print(f"[LEGITIMACY CHECK] Failed - Only {keyword_matches} legal keywords found")
        return False, "This document does not appear to be a legal or court-related document. Please upload a court judgment, order, petition, or other legal document.", "low"
    
    # If many keywords, likely legitimate (skip AI check to save time/quota)
    if keyword_matches >= 10:
        print(f"[LEGITIMACY CHECK] Passed - {keyword_matches} legal keywords found (high confidence)")
        return True, "Document validated as legal/court document", "high"
    
    # Medium confidence - use AI for verification
    print(f"[LEGITIMACY CHECK] {keyword_matches} legal keywords found - using AI verification...")
    sys.stdout.flush()
    
    # Use AI to verify (take first 2000 chars for quick check)
    sample_text = document_text[:2000]
    
    verification_prompt = f"""Analyze this document excerpt and determine if it is a legal or court-related document.

DOCUMENT EXCERPT:
{sample_text}

TASK: Determine if this is a legitimate legal document such as:
- Court judgment or order
- Court petition or application
- Legal notice or summons
- Case filing or pleading
- Legal agreement or contract
- Any other court or legal document

Respond with ONLY ONE WORD:
- "LEGITIMATE" if it is a legal/court document
- "NOT_LEGITIMATE" if it is not a legal/court document

ONE WORD RESPONSE:"""

    try:
        # Check connectivity dynamically
        online_available = is_online(force_check=False)  # Use cached result for speed
        
        if online_available:
            print(f"[LEGITIMACY CHECK] Using online mode (Gemini API)")
            response = call_gemini_with_retry(generation_model, verification_prompt, max_retries=2, initial_wait=20, allow_fallback=True)
            
            if response is None:
                # Fallback was triggered
                print(f"[LEGITIMACY CHECK] Online mode failed, using offline mode")
                result = get_offline_response(verification_prompt, max_tokens=50)
                if result:
                    result = result.strip().upper()
                else:
                    result = None
            else:
                result = response.text.strip().upper()
        else:
            print(f"[LEGITIMACY CHECK] Using offline mode (Ollama/Mistral)")
            result = get_offline_response(verification_prompt, max_tokens=50)
            if result:
                result = result.strip().upper()
            else:
                result = None
        
        if result and "LEGITIMATE" in result and "NOT_LEGITIMATE" not in result:
            print(f"[LEGITIMACY CHECK] Passed - AI confirmed as legal document")
            return True, "Document validated as legal/court document", "medium"
        elif result and "NOT_LEGITIMATE" in result:
            print(f"[LEGITIMACY CHECK] Failed - AI determined not a legal document")
            return False, "This document does not appear to be a legal or court-related document. Our AI analysis indicates this is not a court judgment, order, petition, or other legal document. Please upload a valid legal document for analysis.", "medium"
        else:
            # AI didn't give clear answer, fallback to keyword count
            print(f"[LEGITIMACY CHECK] AI response unclear, using keyword fallback")
            if keyword_matches >= 5:
                print(f"[LEGITIMACY CHECK] Passed - Keyword count ({keyword_matches} keywords)")
                return True, "Document validated as legal/court document (keyword-based)", "medium"
            else:
                print(f"[LEGITIMACY CHECK] Failed - Insufficient legal keywords")
                return False, "Unable to verify document legitimacy. Please ensure you are uploading a legal or court-related document.", "low"
            
    except Exception as e:
        print(f"[LEGITIMACY CHECK] Error during AI verification: {e}")
        # If AI check fails completely, use keyword count as fallback
        if keyword_matches >= 5:
            print(f"[LEGITIMACY CHECK] Passed - Fallback to keyword count ({keyword_matches} keywords)")
            return True, "Document validated as legal/court document (keyword-based)", "medium"
        else:
            print(f"[LEGITIMACY CHECK] Failed - Insufficient legal keywords")
            return False, "Unable to verify document legitimacy. Please ensure you are uploading a legal or court-related document.", "low"

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
    
    # Check connectivity dynamically before each analysis
    use_offline = False
    online_available = is_online(force_check=True)  # Force fresh connectivity check
    
    if online_available:
        print("[ONLINE] MODE: Attempting Gemini API with auto-retry and fallback")
        sys.stdout.flush()
        
        # Prompt 1: Story Summary
        prompt_story = f"""
        Read the following court document. Explain what happened in the case 
        in 1-2 paragraphs, using simple, non-legal language. 
        Describe the events like a straightforward narrative or story.

        DOCUMENT:
        {document_text}
        """
        print("[ONLINE] Generating story summary (auto-retry enabled)...")
        sys.stdout.flush()
        story_response = call_gemini_with_retry(generation_model, prompt_story, allow_fallback=True)
        
        # Check if fallback was triggered
        if story_response is None:
            print("[AUTO-FALLBACK] Story summary failed, switching to offline mode")
            use_offline = True
        else:
            # Prompt 2: Legal Summary
            prompt_legal = f"""
            Act as an expert legal analyst. Read the following court document and
            extract all key legal information. List all cited IPC/CPC sections,
            key dates, and a summary of the most recent actions or judgments.
            Be concise and formal. Output in one single paragraph.

            DOCUMENT:
            {document_text}
            """
            print("[ONLINE] Generating legal summary (auto-retry enabled)...")
            sys.stdout.flush()
            legal_response = call_gemini_with_retry(generation_model, prompt_legal, allow_fallback=True)
            
            # Check if fallback was triggered
            if legal_response is None:
                print("[AUTO-FALLBACK] Legal summary failed, switching to offline mode")
                use_offline = True
            else:
                print("[SUCCESS] Gemini API responded successfully")
                return story_response.text, legal_response.text
    else:
        print("[OFFLINE] MODE: Gemini API not accessible, using offline mode")
        use_offline = True
    
    # Offline mode (either by choice or fallback)
    if use_offline:
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
    Now generates analysis in three stages: legitimacy check, basic analysis, detailed follow-ups.
    Takes legal_summary, story_summary, user_role, and original_doc_text as parameters.
    original_doc_text is used to create a deterministic hash for consistent source retrieval.
    """
    print(f"Generating RAG answer with legitimacy check + two-stage generation (role: {user_role})...")
    sys.stdout.flush()
    
    # --- STAGE 0: LEGITIMACY CHECK ---
    print("[STAGE 0] Checking document legitimacy...")
    sys.stdout.flush()
    
    legitimacy_prompt = f"""You are a legal document validator. Analyze if this is a legitimate legal court document.

DOCUMENT SUMMARY:
{legal_summary[:2000]}

STORY SUMMARY:
{story_summary[:1000]}

TASK: Determine if this is a legitimate legal court document that can be processed in court.

Check for:
1. Is this a court judgment, order, or legal petition?
2. Does it contain proper legal elements (court name, case number, parties, sections, dates)?
3. Is it a valid legal document that can be brought to court?
4. Does it have legal merit and proper structure?

Output ONLY valid JSON with these fields:
{{
  "is_legitimate": true or false,
  "document_type": "court judgment" or "legal petition" or "court order" or "not a legal document",
  "confidence": "high" or "medium" or "low",
  "reason": "Brief explanation (1-2 sentences)"
}}

CRITICAL RULES:
- Output ONLY the JSON object
- Start with {{ and end with }}
- Use true/false (not "true"/"false")
- Be strict: only mark as legitimate if it's clearly a legal court document

Examples:
{{"is_legitimate": true, "document_type": "court judgment", "confidence": "high", "reason": "This is a Supreme Court judgment with proper case details, legal sections, and court decision."}}
{{"is_legitimate": false, "document_type": "not a legal document", "confidence": "high", "reason": "This appears to be a general article or essay, not a court document."}}

Generate JSON:
"""

    try:
        # Check legitimacy using online or offline mode
        legitimacy_raw = None
        if is_online():
            print("[ONLINE] Checking legitimacy with Gemini...")
            sys.stdout.flush()
            try:
                legitimacy_response = call_gemini_with_retry(generation_model, legitimacy_prompt)
                legitimacy_raw = legitimacy_response.text
                print("[SUCCESS] Legitimacy check completed")
            except Exception as e:
                print(f"[ERROR] Gemini failed for legitimacy check: {str(e)[:100]}")
                print("[FALLBACK] Using offline mode for legitimacy check...")
                legitimacy_raw = get_offline_response(legitimacy_prompt, max_tokens=256)
        else:
            print("[OFFLINE] Checking legitimacy with Ollama/Mistral...")
            sys.stdout.flush()
            legitimacy_raw = get_offline_response(legitimacy_prompt, max_tokens=256)
        
        # Parse legitimacy response
        legitimacy_data = {"is_legitimate": True, "document_type": "unknown", "confidence": "low", "reason": "Could not verify"}
        
        if legitimacy_raw:
            # Extract JSON from response
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', legitimacy_raw, re.DOTALL)
            if not match:
                match = re.search(r'\{.*\}', legitimacy_raw, re.DOTALL)
            
            if match:
                try:
                    json_str = match.group(0)
                    json_str = json_str.replace('\n', ' ').replace('\r', '')
                    legitimacy_data = json.loads(json_str)
                    print(f"[LEGITIMACY] Document type: {legitimacy_data.get('document_type', 'unknown')}")
                    print(f"[LEGITIMACY] Is legitimate: {legitimacy_data.get('is_legitimate', False)}")
                    print(f"[LEGITIMACY] Confidence: {legitimacy_data.get('confidence', 'low')}")
                    print(f"[LEGITIMACY] Reason: {legitimacy_data.get('reason', 'N/A')[:100]}")
                    sys.stdout.flush()
                except json.JSONDecodeError as je:
                    print(f"[WARNING] Could not parse legitimacy JSON: {je}")
                    # Assume legitimate if parsing fails (to not break existing functionality)
                    legitimacy_data = {"is_legitimate": True, "document_type": "unknown", "confidence": "low", "reason": "Could not parse legitimacy check"}
        
        # If document is NOT legitimate, return early with explanation
        if not legitimacy_data.get('is_legitimate', True):
            print("[REJECTED] Document is not legitimate, stopping analysis")
            sys.stdout.flush()
            return {
                "is_legitimate": False,
                "document_type": legitimacy_data.get('document_type', 'not a legal document'),
                "confidence": legitimacy_data.get('confidence', 'medium'),
                "legitimacy_reason": legitimacy_data.get('reason', 'This does not appear to be a valid legal court document.'),
                "what_has_happened": "This document cannot be analyzed as it does not appear to be a legitimate legal court document.",
                "key_legal_points": "Analysis not performed - document legitimacy check failed.",
                "general_follow_ups": [
                    f"Document Type: {legitimacy_data.get('document_type', 'Not a legal document')}",
                    f"Reason: {legitimacy_data.get('reason', 'This does not appear to be a valid court judgment, order, or legal petition.')}",
                    "Recommendation: Please upload a valid legal court document such as a court judgment, court order, or legal petition with proper case details, parties, legal sections, and court information.",
                    "What to check: Ensure the document contains court name, case number, parties involved, legal sections cited, dates, and judicial decisions or orders.",
                    "If you believe this is an error: The document may be incomplete, poorly scanned, or in an unsupported format. Try uploading a clearer copy or a different document."
                ],
                "disclaimer": "This document was rejected during legitimacy verification. It does not appear to be a valid legal court document that can be processed."
            }, []
        
        # Document is legitimate, continue with normal analysis
        print("[APPROVED] Document is legitimate, proceeding with full analysis")
        sys.stdout.flush()
        
    except Exception as e:
        print(f"[ERROR] Legitimacy check failed: {e}")
        print("[WARNING] Proceeding with analysis (assuming legitimate)")
        sys.stdout.flush()
        # Continue with analysis if legitimacy check fails (to not break existing functionality)
    
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
        basic_raw_text = None
        use_offline_stage1 = False
        
        if is_online():
            print("[ONLINE] Attempting Gemini API for Stage 1 (with auto-retry)...")
            sys.stdout.flush()
            try:
                basic_response = call_gemini_with_retry(generation_model, basic_prompt)
                basic_raw_text = basic_response.text
                print("[SUCCESS] Gemini API responded for Stage 1")
                # Log if response was blocked
                if hasattr(basic_response, 'prompt_feedback'):
                    print(f"Prompt feedback: {basic_response.prompt_feedback}")
            except Exception as e:
                error_msg = str(e)
                print(f"[ERROR] Gemini API failed for Stage 1 after retries: {error_msg[:200]}")
                if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                    print("[AUTO-FALLBACK] Gemini quota still exceeded after retries, switching to offline for Stage 1...")
                    use_offline_stage1 = True
                else:
                    print("[AUTO-FALLBACK] Gemini error, switching to offline for Stage 1...")
                    use_offline_stage1 = True
        else:
            use_offline_stage1 = True
        
        if use_offline_stage1 or not basic_raw_text:
            print("[OFFLINE] Using Ollama/Mistral for Stage 1...")
            basic_raw_text = get_offline_response(basic_prompt, max_tokens=1024)
            if not basic_raw_text:
                print("[ERROR] Offline model not available.")
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

        followup_raw_text = None
        use_offline_stage2 = False
        
        if is_online():
            print("[ONLINE] Attempting Gemini API for Stage 2 (with auto-retry)...")
            sys.stdout.flush()
            try:
                followup_response = call_gemini_with_retry(generation_model, followup_prompt)
                followup_raw_text = followup_response.text
                print(f"[SUCCESS] Gemini API responded for Stage 2 ({len(followup_raw_text)} chars)")
                # Log if response was blocked
                if hasattr(followup_response, 'prompt_feedback'):
                    print(f"Prompt feedback: {followup_response.prompt_feedback}")
            except Exception as e:
                error_msg = str(e)
                print(f"[ERROR] Gemini API failed for Stage 2 after retries: {error_msg[:200]}")
                if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                    print("[AUTO-FALLBACK] Gemini quota still exceeded after retries, switching to offline for Stage 2...")
                    use_offline_stage2 = True
                else:
                    print("[AUTO-FALLBACK] Gemini error, switching to offline for Stage 2...")
                    use_offline_stage2 = True
        else:
            use_offline_stage2 = True
        
        if use_offline_stage2 or not followup_raw_text:
            print("[OFFLINE] Using Ollama/Mistral for Stage 2...")
            followup_raw_text = get_offline_response(followup_prompt, max_tokens=2048)
            if not followup_raw_text:
                print("[ERROR] Could not generate follow-ups")
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
    """Handles the file upload and analysis.
    
    OWASP Security: File validation, size limits, secure filename, path traversal prevention, content validation
    """
    # SECURITY: Validate file upload (A03:2025 - Injection)
    print(f"[DEBUG] Analyze request received")
    print(f"[DEBUG] Request files: {list(request.files.keys())}")
    print(f"[DEBUG] Request form: {dict(request.form)}")
    
    if 'file' not in request.files:
        print(f"[ERROR] No file part in request")
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    user_role = sanitize_input(request.form.get('user_role', 'unknown'), max_length=20)
    
    print(f"[DEBUG] File received: {file.filename}")
    print(f"[DEBUG] User role: {user_role}")
    
    if file.filename == '':
        print(f"[ERROR] Empty filename")
        return jsonify({"error": "No selected file"}), 400
    
    # SECURITY: Validate user role (A03:2025 - Injection)
    allowed_roles = ['plaintiff', 'defendant', 'unknown']
    if user_role not in allowed_roles:
        return jsonify({"error": "Invalid role selection"}), 400
    
    if not user_role or user_role == '':
        return jsonify({"error": "Please select your role in the case"}), 400
    
    # SECURITY: Validate file extension (A03:2025 - Injection)
    if not allowed_file(file.filename):
        print(f"[SECURITY] File rejected - invalid extension: {file.filename}")
        return jsonify({"error": "Invalid file type. Only PDF files are allowed."}), 400
    
    # SECURITY: Validate file size (A04:2025 - Insecure Design - DoS prevention)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset file pointer
    
    if file_size == 0:
        print(f"[SECURITY] File rejected - empty file")
        return jsonify({"error": "File is empty"}), 400
    
    if file_size > app.config['MAX_CONTENT_LENGTH']:
        print(f"[SECURITY] File rejected - too large: {file_size} bytes")
        return jsonify({"error": f"File too large. Maximum size is {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB"}), 400
    
    # SECURITY: Secure filename to prevent path traversal (A03:2025 - Injection)
    original_filename = file.filename
    safe_filename = validate_filename(original_filename)
    if not safe_filename:
        print(f"[SECURITY] File rejected - invalid filename: {original_filename}")
        return jsonify({"error": "Invalid filename. Please rename your file."}), 400
    
    # SECURITY: Add timestamp and random component to prevent filename collisions and prediction
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    random_suffix = secrets.token_hex(4)
    unique_filename = f"{timestamp}_{random_suffix}_{safe_filename}"
    
    # SECURITY: Use secure path join to prevent directory traversal (A03:2025 - Injection)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    filepath = os.path.abspath(filepath)  # Resolve to absolute path
    
    # SECURITY: Verify file is within upload folder (A01:2025 - Broken Access Control)
    upload_folder_abs = os.path.abspath(app.config['UPLOAD_FOLDER'])
    if not filepath.startswith(upload_folder_abs):
        print(f"[SECURITY] File rejected - path traversal attempt")
        return jsonify({"error": "Invalid file path"}), 400
    
    # SECURITY: Verify PDF magic bytes (A03:2025 - Injection)
    try:
        file_header = file.read(4)
        file.seek(0)  # Reset file pointer to beginning
        
        if file_header != b'%PDF':
            print(f"[SECURITY] File rejected - invalid PDF magic bytes: {file_header}")
            return jsonify({"error": "Invalid PDF file. File does not appear to be a valid PDF."}), 400
        
        print(f"[OK] PDF magic bytes verified: {file_header}")
    except Exception as e:
        print(f"[ERROR] Failed to read file header: {e}")
        return jsonify({"error": "Failed to validate file"}), 400
    
    # Save the uploaded file
    try:
        file.save(filepath)
        
        # SECURITY: Verify file was saved correctly (A08:2025 - Software Integrity Failures)
        if not os.path.exists(filepath):
            return jsonify({"error": "Failed to save file"}), 500
        
        # SECURITY: Verify saved file size matches uploaded file size
        saved_size = os.path.getsize(filepath)
        if saved_size != file_size:
            os.remove(filepath)
            return jsonify({"error": "File integrity check failed"}), 500
    except Exception as e:
        print(f"[ERROR] Failed to save file: {e}")
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        return jsonify({"error": "Failed to save file"}), 500
    
    # File saved successfully, proceed with analysis
    print(f"\n{'='*60}")
    print(f"--- New Job Started for {file.filename} ---")
    print(f"[INFO] User role: {user_role.upper()}")
    print(f"[DEBUG] File saved to: {filepath}")
    sys.stdout.flush()  # Force output immediately
    
    # --- Run the full pipeline ---
    
    # 1. Read PDF
    print("[STEP 1/4] Reading PDF...")
    sys.stdout.flush()
    doc_text = extract_text_from_pdf(filepath)
    if not doc_text:
        print("[ERROR] Could not extract text from PDF")
        sys.stdout.flush()
        return jsonify({"error": "Could not read text from PDF."}), 500
    print(f"[OK] Extracted {len(doc_text)} characters from PDF")
    sys.stdout.flush()
    
    # 2. Legitimacy Check
    print("[STEP 2/4] Checking document legitimacy...")
    sys.stdout.flush()
    is_legitimate, legitimacy_message, confidence = check_document_legitimacy(doc_text)
    
    if not is_legitimate:
        print(f"[REJECTED] Document failed legitimacy check: {legitimacy_message}")
        sys.stdout.flush()
        return jsonify({
            "error": "Document Not Accepted",
            "legitimacy_check": {
                "passed": False,
                "message": legitimacy_message,
                "confidence": confidence
            }
        }), 400
    
    print(f"[PASSED] Document legitimacy check passed ({confidence} confidence)")
    sys.stdout.flush()
    
    # 3. Get Summaries
    print("[STEP 3/4] Generating summaries...")
    sys.stdout.flush()
    story_summary, legal_summary = get_summaries_from_text(doc_text)
    print(f"[OK] Story summary: {len(story_summary)} chars")
    print(f"[OK] Legal summary: {len(legal_summary)} chars")
    sys.stdout.flush()
    
    # 4. Get RAG Analysis (now with two-stage generation)
    print("[STEP 4/4] Generating RAG analysis...")
    sys.stdout.flush()
    analysis_data, sources = get_rag_answer(legal_summary, story_summary, user_role, doc_text)
    
    print("="*60)
    print("--- Job Complete ---")
    print(f"[RESULT] Analysis fields: {list(analysis_data.keys())}")
    print(f"[RESULT] Sources count: {len(sources)}")
    print(f"[RESULT] Final sources: {sources}")
    print("="*60)
    sys.stdout.flush()
    
    # 5. Auto-save analysis for logged-in users
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

    # 6. Return all results as JSON
    return jsonify({
        "legitimacy_check": {
            "passed": True,
            "message": legitimacy_message,
            "confidence": confidence
        },
        "storySummary": story_summary,
        "legalSummary": legal_summary,
        "analysis": analysis_data,  # <-- This is now an object, not a string
        "sources": sources,
        "recommended": {'category': detect_case_category(analysis_data, legal_summary), 'recommended': SAMPLE_LAWYERS.get(detect_case_category(analysis_data, legal_summary), SAMPLE_LAWYERS['other'])}
    })

@app.route('/get-file/<path:filepath>')
def get_file(filepath):
    """
    Serves a file from the BASE_DIR.
    This is used to make the source links clickable.
    
    OWASP Security: Path traversal prevention, file access control, content type validation
    """
    # SECURITY: Validate filepath parameter (A01:2025 - Broken Access Control)
    if not filepath:
        return "Invalid file path", 400
    
    # SECURITY: Sanitize filepath to prevent path traversal (A01:2025 - Broken Access Control)
    if '..' in filepath or filepath.startswith('/') or filepath.startswith('\\'):
        print(f"[SECURITY] Path traversal attempt blocked: {filepath}")
        return "Access denied", 403
    
    # SECURITY: Only allow access to specific directories (A01:2025 - Broken Access Control)
    allowed_dirs = ['supreme_court_judgments', 'ipc_sections', 'cpc_sections', 'constitution']
    if not any(filepath.startswith(allowed_dir + '/') or filepath.startswith(allowed_dir + '\\') for allowed_dir in allowed_dirs):
        print(f"[SECURITY] Access to unauthorized directory blocked: {filepath}")
        return "Access denied", 403
    
    # SECURITY: Construct and validate full path (A01:2025 - Broken Access Control)
    try:
        full_path = os.path.join(BASE_DIR, filepath)
        full_path = os.path.abspath(full_path)  # Resolve to absolute path
        base_dir_abs = os.path.abspath(BASE_DIR)
        
        # SECURITY: Ensure file is within allowed base directory (A01:2025 - Broken Access Control)
        if not full_path.startswith(base_dir_abs):
            print(f"[SECURITY] Path traversal attempt blocked: {filepath}")
            return "Access denied", 403
        
        # SECURITY: Only serve PDF files (A01:2025 - Broken Access Control)
        if not full_path.lower().endswith('.pdf'):
            print(f"[SECURITY] Non-PDF file access blocked: {filepath}")
            return "Invalid file type", 400
        
        # SECURITY: Check if file exists and is a file (not directory)
        if not os.path.exists(full_path) or not os.path.isfile(full_path):
            return "File not found", 404
        
        # SECURITY: Verify file size is reasonable (prevent serving huge files)
        file_size = os.path.getsize(full_path)
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            print(f"[SECURITY] File too large: {filepath} ({file_size} bytes)")
            return "File too large", 413
        
        # SECURITY: Verify PDF magic bytes (A08:2025 - Software Integrity Failures)
        with open(full_path, 'rb') as f:
            file_header = f.read(4)
            if file_header != b'%PDF':
                print(f"[SECURITY] Invalid PDF file: {filepath}")
                return "Invalid file", 400
        
        print(f"[OK] Serving file: {filepath}")
        
        # SECURITY: Serve file with security headers (A05:2025 - Security Misconfiguration)
        directory = os.path.dirname(full_path)
        filename = os.path.basename(full_path)
        response = send_from_directory(directory, filename, as_attachment=False)
        
        # SECURITY: Add security headers for file download (A05:2025 - Security Misconfiguration)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Disposition'] = f'inline; filename="{filename}"'
        response.headers['Content-Type'] = 'application/pdf'
        
        return response
        
    except Exception as e:
        print(f"[ERROR] Error serving file {filepath}: {e}")
        return "Error serving file", 500


@app.route('/register', methods=['POST'])
def register():
    """Registers a new user. Expects JSON with profile fields.
    Required: username, password
    Optional: full_name, phone, address_current, address_permanent, dob, email
    
    OWASP Security: Input validation, XSS prevention, SQL injection prevention, password strength
    """
    # SECURITY: Validate CSRF for state-changing operation (A01:2025 - Broken Access Control)
    if not validate_csrf_token():
        return jsonify({'error': 'Invalid request origin'}), 403
    
    # Accept JSON API or standard form POST
    data = request.get_json(silent=True)
    if not data:
        data = request.form

    # SECURITY: Sanitize all inputs (A03:2025 - Injection)
    username = sanitize_input((data.get('username') or '').strip(), max_length=50)
    password = data.get('password') or ''
    full_name = sanitize_input((data.get('full_name') or '').strip(), max_length=100)
    phone = sanitize_input((data.get('phone') or '').strip(), max_length=20)
    address_current = sanitize_input((data.get('address_current') or '').strip(), max_length=500)
    address_permanent = sanitize_input((data.get('address_permanent') or '').strip(), max_length=500)
    dob = sanitize_input((data.get('dob') or '').strip(), max_length=10)
    email = sanitize_input((data.get('email') or '').strip(), max_length=254)

    # SECURITY: Validate required fields
    if not username or not password:
        return jsonify({'error': 'Username and password are required.'}), 400
    
    # SECURITY: Validate username format (A03:2025 - Injection)
    if not validate_username(username):
        return jsonify({'error': 'Username must be 3-50 characters (letters, numbers, underscore only).'}), 400
    
    # SECURITY: Validate password strength (A07:2025 - Auth Failures)
    is_strong, message = validate_password_strength(password)
    if not is_strong:
        return jsonify({'error': message}), 400
    
    # SECURITY: Validate optional fields (A03:2025 - Injection)
    if email and not validate_email(email):
        return jsonify({'error': 'Invalid email format.'}), 400
    
    if phone and not validate_phone(phone):
        return jsonify({'error': 'Invalid phone number format.'}), 400
    
    # SECURITY: Validate date of birth format (A03:2025 - Injection)
    if dob:
        try:
            # Accept YYYY-MM-DD format
            datetime.strptime(dob, '%Y-%m-%d')
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD.'}), 400
    
    # SECURITY: Hash password with strong algorithm (A02:2025 - Cryptographic Failures)
    pwd_hash = generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # SECURITY: Check if username already exists (prevent timing attacks)
        cur.execute('SELECT id FROM users WHERE username = ?', (username,))
        existing_user = cur.fetchone()
        
        if existing_user:
            conn.close()
            # SECURITY: Generic error message to prevent user enumeration (A07:2025 - Auth Failures)
            return jsonify({'error': 'Username already exists.'}), 400
        
        # SECURITY: Parameterized query to prevent SQL injection (A03:2025 - Injection)
        cur.execute('''INSERT INTO users (username, password_hash, created_at, full_name, phone, address_current, address_permanent, dob, email)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (username, pwd_hash, datetime.utcnow().isoformat(), full_name, phone, address_current, address_permanent, dob, email))
        conn.commit()
        user_id = cur.lastrowid
        conn.close()
        
        # SECURITY: Log successful registration (A09:2025 - Security Logging)
        log_security_event('REGISTRATION_SUCCESS', f'New user registered: {username}', get_client_ip(), user_id)
        
        # SECURITY: Regenerate session ID to prevent session fixation (A07:2025 - Auth Failures)
        session.clear()
        session['user_id'] = user_id
        session['username'] = username
        session.permanent = True
        
        # If original request was a form submit, redirect to dashboard
        if not request.is_json:
            return redirect(url_for('dashboard_page'))
        return jsonify({'ok': True, 'username': username})
        
    except sqlite3.IntegrityError:
        # SECURITY: Log registration attempt with existing username (A09:2025 - Security Logging)
        log_security_event('REGISTRATION_FAILED', f'Attempt to register existing username: {username}', get_client_ip(), severity='WARNING')
        return jsonify({'error': 'Username already exists.'}), 400
    except Exception as e:
        print(f"Error registering user: {e}")
        log_security_event('REGISTRATION_ERROR', f'Registration error: {str(e)}', get_client_ip(), severity='ERROR')
        return jsonify({'error': 'An error occurred during registration. Please try again.'}), 500


@app.route('/login', methods=['POST'])
def login():
    """Logs a user in. Expects JSON: {username, password}
    
    OWASP Security: Rate limiting, timing attack prevention, secure sessions, CSRF protection
    """
    # SECURITY: Validate CSRF for state-changing operation (A01:2025 - Broken Access Control)
    if not validate_csrf_token():
        return jsonify({'error': 'Invalid request origin'}), 403
    
    # SECURITY: Get client IP for rate limiting (A07:2025 - Auth Failures)
    client_ip = get_client_ip()
    
    # SECURITY: Check rate limit (A07:2025 - Auth Failures)
    if not check_rate_limit(client_ip):
        return jsonify({'error': 'Too many login attempts. Please try again in 15 minutes.'}), 429
    
    data = request.get_json(silent=True)
    if not data:
        data = request.form
    
    # SECURITY: Sanitize inputs (A03:2025 - Injection)
    username = sanitize_input((data.get('username') or '').strip(), max_length=50)
    password = data.get('password') or ''

    if not username or not password:
        # SECURITY: Record failed attempt
        record_login_attempt(client_ip, success=False)
        return jsonify({'error': 'Username and password are required.'}), 400
    
    # SECURITY: Validate username format (A03:2025 - Injection)
    if not validate_username(username):
        # SECURITY: Record failed attempt
        record_login_attempt(client_ip, success=False)
        return jsonify({'error': 'Invalid credentials.'}), 401
    
    # SECURITY: Validate password length (prevent DoS with huge passwords)
    if len(password) > 128:
        # SECURITY: Record failed attempt
        record_login_attempt(client_ip, success=False)
        return jsonify({'error': 'Invalid credentials.'}), 401

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # SECURITY: Parameterized query prevents SQL injection (A03:2025 - Injection)
        cur.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,))
        row = cur.fetchone()
        conn.close()
        
        # SECURITY: Constant-time comparison to prevent timing attacks (A07:2025 - Auth Failures)
        if row and check_password_hash(row['password_hash'], password):
            # SECURITY: Record successful login
            record_login_attempt(client_ip, success=True)
            
            # SECURITY: Log successful login (A09:2025 - Security Logging)
            log_security_event('LOGIN_SUCCESS', f'User {username} logged in successfully', client_ip, row['id'])
            
            # SECURITY: Regenerate session ID to prevent session fixation (A07:2025 - Auth Failures)
            session.clear()
            session['user_id'] = row['id']
            session['username'] = username
            session.permanent = True  # Use PERMANENT_SESSION_LIFETIME
            
            if not request.is_json:
                return redirect(url_for('dashboard_page'))
            return jsonify({'ok': True, 'username': username})
        else:
            # SECURITY: Record failed login attempt
            record_login_attempt(client_ip, success=False)
            
            # SECURITY: Log failed login attempt (A09:2025 - Security Logging)
            log_security_event('LOGIN_FAILED', f'Failed login attempt for username: {username}', client_ip, severity='WARNING')
            
            # SECURITY: Generic error message to prevent user enumeration (A07:2025 - Auth Failures)
            return jsonify({'error': 'Invalid credentials.'}), 401
    except Exception as e:
        print(f"Error during login: {e}")
        log_security_event('LOGIN_ERROR', f'Login error: {str(e)}', get_client_ip(), severity='ERROR')
        return jsonify({'error': 'An error occurred during login. Please try again.'}), 500


@app.route('/logout', methods=['POST'])
def logout():
    """Logs out the current user.
    
    OWASP Security: Secure session termination, logging
    """
    user_id = session.get('user_id')
    username = session.get('username')
    
    # SECURITY: Log logout event (A09:2025 - Security Logging)
    if user_id:
        log_security_event('LOGOUT', f'User {username} logged out', get_client_ip(), user_id)
    
    # SECURITY: Clear all session data (A07:2025 - Auth Failures)
    session.clear()
    
    return jsonify({'ok': True})


@app.route('/my_analyses', methods=['GET'])
@require_auth  # SECURITY: Require authentication (A01:2025 - Broken Access Control)
def my_analyses():
    """Returns list of analyses for the logged-in user.
    
    OWASP Security: Authentication required, user isolation
    """

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

# --- OWASP SECURITY: ERROR HANDLERS (A05:2025 - Security Misconfiguration) ---

@app.errorhandler(400)
def bad_request(error):
    """Handle bad request errors without exposing details"""
    return jsonify({'error': 'Bad request'}), 400

@app.errorhandler(401)
def unauthorized(error):
    """Handle unauthorized access"""
    return jsonify({'error': 'Authentication required'}), 401

@app.errorhandler(403)
def forbidden(error):
    """Handle forbidden access"""
    return jsonify({'error': 'Access denied'}), 403

@app.errorhandler(404)
def not_found(error):
    """Handle not found errors"""
    if request.path.startswith('/api/') or request.is_json:
        return jsonify({'error': 'Resource not found'}), 404
    return render_template('404.html'), 404

@app.errorhandler(405)
def method_not_allowed(error):
    """Handle method not allowed errors"""
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors"""
    return jsonify({'error': 'File too large. Maximum size is 50MB'}), 413

@app.errorhandler(429)
def too_many_requests(error):
    """Handle rate limit errors"""
    return jsonify({'error': 'Too many requests. Please try again later'}), 429

@app.errorhandler(500)
def internal_server_error(error):
    """Handle internal server errors without exposing details"""
    # Log the actual error for debugging
    print(f"[ERROR] Internal server error: {error}")
    # Return generic error to user (prevent information disclosure)
    return jsonify({'error': 'An internal error occurred. Please try again later'}), 500

@app.errorhandler(Exception)
def handle_exception(error):
    """Handle all unhandled exceptions"""
    # Log the actual error for debugging
    print(f"[ERROR] Unhandled exception: {type(error).__name__}: {error}")
    
    # Return generic error to user (prevent information disclosure)
    if request.path.startswith('/api/') or request.is_json:
        return jsonify({'error': 'An error occurred. Please try again later'}), 500
    return render_template('error.html'), 500

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