# CodeBridge

CodeBridge is a multi-language code runner and transpiler with a web UI.

Supported languages:
- Python
- JavaScript
- Java
- C++
- Ruby

## Features
- Run code with optional stdin input.
- Convert code between supported languages.
- Compile/check syntax and show errors in UI.
- Analyze code and fetch language templates via API.

## Project Structure
```text
code_bridge/
  backend/
    app.py
    requirements.txt
    compilers/
    transpiler/
  frontend/
    index.html
    css/styles.css
    js/app.js
  run_compiler.bat
  README.md
  .gitignore
```

## Prerequisites
Install and ensure these are in PATH:
- Python 3.10+
- Node.js
- Java JDK
- g++ (MinGW or similar)
- Ruby

Check versions:
```bash
python --version
node --version
java --version
g++ --version
ruby --version
```

## Setup
```bash
cd code_bridge
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Option 1 (Windows helper script):
```bat
run_compiler.bat
```

Option 2 (manual):
```bash
# Terminal 1
cd backend
python app.py

# Terminal 2
cd frontend
python -m http.server 8080
```

Open:
- Frontend: http://localhost:8080
- Backend: http://localhost:5000

## API Endpoints
- GET `/`
- GET `/api/languages`
- POST `/api/execute`
- POST `/api/convert`
- POST `/api/compile`
- POST `/api/analyze`
- GET `/api/templates`

Example execute request:
```json
{
  "code": "print('Hello')",
  "language": "python",
  "stdin": ""
}
```

## GitHub Push
```bash
git init
git add .
git commit -m "Initial commit: CodeBridge"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## Notes
- Generated artifacts are ignored by `.gitignore`.
- If a language fails to run, verify its compiler/interpreter installation and PATH.
