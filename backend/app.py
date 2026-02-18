"""
CodeBridge - Multi-Backend Compiler Framework
Main Flask Application
"""

import os
import sys
import json
import subprocess
import tempfile
import threading
import uuid
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Import compiler modules
from compilers.python_compiler import PythonCompiler
from compilers.java_compiler import JavaCompiler
from compilers.cpp_compiler import CPPCompiler
from compilers.javascript_compiler import JavaScriptCompiler
from compilers.ruby_compiler import RubyCompiler
from transpiler.code_transpiler import CodeTranspiler

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['TIMEOUT'] = 30  # 30 seconds timeout for code execution

# Initialize compilers
compilers = {
    'python': PythonCompiler(),
    'java': JavaCompiler(),
    'cpp': CPPCompiler(),
    'javascript': JavaScriptCompiler(),
    'ruby': RubyCompiler()
}

# Initialize transpiler
transpiler = CodeTranspiler()

# Store execution results
execution_results = {}


def get_unique_id():
    """Generate a unique ID for each execution"""
    return str(uuid.uuid4())


@app.route('/', methods=['GET'])
def index():
    """Health check endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'CodeBridge API is running',
        'version': '1.0.0',
        'supported_languages': list(compilers.keys())
    })


@app.route('/api/languages', methods=['GET'])
def get_languages():
    """Get list of supported languages"""
    languages = []
    for lang, compiler in compilers.items():
        languages.append({
            'name': lang,
            'display_name': compiler.get_display_name(),
            'extension': compiler.get_extension(),
            'version': compiler.get_version()
        })
    return jsonify({
        'status': 'success',
        'languages': languages
    })


@app.route('/api/execute', methods=['POST'])
def execute_code():
    """Execute code in the specified language"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        code = data.get('code', '')
        language = data.get('language', '').lower()
        stdin_input = data.get('stdin', '')
        
        if not code:
            return jsonify({
                'status': 'error',
                'message': 'No code provided'
            }), 400
        
        if language not in compilers:
            return jsonify({
                'status': 'error',
                'message': f'Unsupported language: {language}. Supported languages: {list(compilers.keys())}'
            }), 400
        
        # Execute the code
        compiler = compilers[language]
        result = compiler.execute(code, stdin_input, timeout=app.config['TIMEOUT'])
        
        return jsonify({
            'status': 'success',
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/convert', methods=['POST'])
def convert_code():
    """Convert code from one language to another"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        code = data.get('code', '')
        source_language = data.get('source_language', '').lower()
        target_language = data.get('target_language', '').lower()
        
        if not code:
            return jsonify({
                'status': 'error',
                'message': 'No code provided'
            }), 400
        
        if source_language not in compilers:
            return jsonify({
                'status': 'error',
                'message': f'Unsupported source language: {source_language}'
            }), 400
        
        if target_language not in compilers:
            return jsonify({
                'status': 'error',
                'message': f'Unsupported target language: {target_language}'
            }), 400
        
        if source_language == target_language:
            return jsonify({
                'status': 'success',
                'result': {
                    'original_code': code,
                    'converted_code': code,
                    'source_language': source_language,
                    'target_language': target_language,
                    'message': 'Source and target languages are the same'
                }
            })
        
        # Convert the code
        result = transpiler.transpile(code, source_language, target_language)
        
        return jsonify({
            'status': 'success',
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/compile', methods=['POST'])
def compile_code():
    """Compile code and return any errors (for languages that support separate compilation)"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        code = data.get('code', '')
        language = data.get('language', '').lower()
        
        if not code:
            return jsonify({
                'status': 'error',
                'message': 'No code provided'
            }), 400
        
        if language not in compilers:
            return jsonify({
                'status': 'error',
                'message': f'Unsupported language: {language}'
            }), 400
        
        compiler = compilers[language]
        
        # Check if compiler supports compilation
        if hasattr(compiler, 'compile'):
            result = compiler.compile(code)
        else:
            # For interpreted languages, just check syntax
            if hasattr(compiler, 'check_syntax'):
                result = compiler.check_syntax(code)
            else:
                result = {
                    'success': True,
                    'message': 'Language does not require compilation'
                }
        
        return jsonify({
            'status': 'success',
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_code():
    """Analyze code and return structure information"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'No JSON data provided'
            }), 400
        
        code = data.get('code', '')
        language = data.get('language', '').lower()
        
        if not code:
            return jsonify({
                'status': 'error',
                'message': 'No code provided'
            }), 400
        
        if language not in compilers:
            return jsonify({
                'status': 'error',
                'message': f'Unsupported language: {language}'
            }), 400
        
        compiler = compilers[language]
        
        # Analyze the code
        if hasattr(compiler, 'analyze'):
            result = compiler.analyze(code)
        else:
            result = {
                'lines': len(code.split('\n')),
                'characters': len(code),
                'language': language
            }
        
        return jsonify({
            'status': 'success',
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/templates', methods=['GET'])
def get_templates():
    """Get code templates for each language"""
    templates = {}
    for lang, compiler in compilers.items():
        if hasattr(compiler, 'get_template'):
            templates[lang] = compiler.get_template()
    
    return jsonify({
        'status': 'success',
        'templates': templates
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500


@app.errorhandler(413)
def too_large(error):
    return jsonify({
        'status': 'error',
        'message': 'File too large. Maximum size is 16MB'
    }), 413


if __name__ == '__main__':
    # Create temp directory if it doesn't exist
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    print("=" * 50)
    print("CodeBridge - Multi-Backend Compiler Framework")
    print("=" * 50)
    print(f"Server running on http://localhost:5000")
    print(f"Supported languages: {list(compilers.keys())}")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5000, debug=True)