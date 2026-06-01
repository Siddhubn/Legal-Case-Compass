# Legal Case Compass (LCC): AI Legal Navigator for Litigants ⚖️🚀

Legal Case Compass (LCC) is an enhanced, AI-assisted Legal Intelligent system engineered to bridge the "comprehension gap" in the Indian judicial system. The platform processes unstructured legal PDFs—such as court orders, petitions, and notices—and converts them into verified, plain-language insights and role-specific strategic roadmaps, making justice visually transparent and legally accessible.

---

## 📸 Application Snapshots

| **Secure User Authentication** | **Main Dashboard & Upload Zone** |
|---|---|
| ![Secure User Login](placeholders/snapshot_login.png) <br> *Figure 1: Secure authentication interface using unique Navy/Gold judiciary styling.* | ![Main Dashboard](placeholders/snapshot_dashboard.png) <br> *Figure 2: Drag-and-drop area accepting documents under 16MB.* |

| **"Traffic Light" Verification & Summarization** | **Strategic Advice & Precedent Mapping** |
|---|---|
| ![Analysis Results](placeholders/snapshot_traffic_light.png) <br> *Figure 3: Immediate legitimacy check banner with a tabbed view for Plain Language vs. Technical summaries.* | ![Strategic Recommendations](placeholders/snapshot_recommendations.png) <br> *Figure 4: Role-based actionable advice paired with verifiable Supreme Court precedents.* |

---

## ✨ Core Features & Analysis Pipeline

1. **Document Legitimacy Verification ("Traffic Light" System):** A rule-based scoring module utilizing 19+ unique identifiers to check file structure. It buckets uploads instantly into:
   * 🟢 **Legally Runnable:** Genuine legal document, safe for pipeline analysis.
   * 🟡 **Questionable:** Missing procedural attributes but can proceed with caution.
   * 🔴 **Not Runnable:** Halts non-legal files immediately to protect compute power.
2. **Dual-Perspective Summarization Engine:** Generates a story-styled **Plain-Language Summary** for laypersons to strip out archaic jargon, alongside a structured **Technical Legal Summary** highlighting statutory clauses and procedural postures for professionals.
3. **Precedent-Grounded RAG Pipeline:** Generates conceptual vector queries matched against a local **ChromaDB** index containing **10,000+ landmark Supreme Court of India judgments**. This provides high transparency and forces a near-zero hallucination rate.
4. **Deterministic Search Logic:** Employs binary text caching via content-based file hashing. If the same document is re-uploaded, the system returns matching results immediately, guaranteeing operational consistency.
5. **Hybrid Inference Engine (Offline Fallback):** Intelligently monitors API state. It uses cloud-based **Google Gemini 1.5 Flash** for lightning-fast analysis, but executes a seamless **graceful degradation to local Ollama (Llama3/Mistral)** instances if internet connectivity drops.

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| **Backend Framework** | Flask (Python 3.10+) | Server routing, session security, API orchestration |
| **Cloud Intelligence** | Google Gemini 1.5 Flash API | High-fidelity legal summary & strategy synthesis |
| **Offline Inference** | Ollama (Llama3 / Mistral) | Edge model computing for zero-network environments |
| **Vector DB (RAG)** | ChromaDB | In-process high-dimensional similarity index storage |
| **Text Processing** | PyMuPDF (Fitz) & Regex | Layout-aware string extraction from binary stream |
| **UI Design** | Tailwind CSS & Jinja2 | Responsive layout styled on classic digital courtrooms |
| **Relational DB** | SQLite (SQLAlchemy) | Secure transactional audit trails & user profiling |

---

## 🔒 Security Configuration (OWASP Compliance)

* **Magic Byte Extraction:** Validates binary headers strictly for the explicit `%PDF-` ASCII sequence signature to block disguised executables (`.exe`).
* **Path Traversal Shield:** Filenames are automatically sanitized (`secure_filename`) and stamped with automated random tokens.
* **XSS & SQL Injection Mitigation:** Uses the `Bleach` library for absolute string sanitization and parameterized query bindings through SQLAlchemy to protect active user spaces.
* **Session Security:** Passwords are fully encrypted using Werkzeug PBKDF2 salts; application session cookies rely on `HttpOnly` and `SameSite=Lax` tags.

---

## ⚙️ Local Setup Instructions

### Prerequisites
* Python 3.10 or higher
* Ollama installed and running locally (`ollama run llama3`)

### 1. Clone & Environment Isolation
```bash
git clone [https://github.com/yourusername/legal-case-compass.git](https://github.com/yourusername/legal-case-compass.git)
cd legal-case-compass

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Environment Variables

Create a configuration file named `.env` in your root workspace:

```ini
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your_secure_session_token_here
GOOGLE_API_KEY=your_gemini_api_key_here
CHROMA_DB_PATH=./chroma_db

```

### 4. Build the Local Vector Store & Run

```bash
# Optional setup step depending on how your data script triggers index creation
python build_vector_db.py  

# Execute the local application web server
flask run

```

Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## ⚖️ Legal Disclaimer

*Legal Case Compass (LCC) is designed purely as an informational, preliminary aid to decode legal documentation and enhance visibility into case workflows. It does not generate certified legal advice and must not serve as an direct substitute for professional human counsel or qualified attorneys.*

```

***

### 💡 Tips for Completing Your Repository Setup:
1. [cite_start]**Placeholder Folders:** Create a directory called `placeholders/` or `assets/` inside your Git tree, and put the design snapshots from your implementation there[cite: 2164].
2. **Naming Images:** Ensure the file names of your saved screenshots match the path string inside the brackets exactly (e.g., replace `placeholders/snapshot_login.png` with your actual file path). 
3. [cite_start]**Quantized Footprint Reminder:** As evaluated in your benchmarking suite, remind users running your repo locally that a minimum of **8GB RAM** is required for local processing to prevent system memory bottlenecks when cloud fallback mode activates[cite: 2216, 2221].