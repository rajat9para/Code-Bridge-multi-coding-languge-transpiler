"""
CodeBridge - JavaScript Compiler
"""

import os
import re
import html
import unicodedata
from .base_compiler import BaseCompiler


class JavaScriptCompiler(BaseCompiler):
    """JavaScript language compiler and executor using Node.js"""
    
    def __init__(self):
        super().__init__()
        self.language_name = "javascript"
        self.display_name = "JavaScript"
        self.extension = ".js"
        self.version_command = ["node", "--version"]
    
    def execute(self, code, stdin_input="", timeout=30):
        """Execute JavaScript code using Node.js"""
        code = self._normalize_javascript_code(code)

        # Create a temporary file
        filepath = self._create_temp_file(code)
        
        try:
            # Fail early with clear syntax errors after normalization
            syntax_check = self._run_command(
                ["node", "--check", filepath],
                timeout=min(timeout, 10)
            )
            if not syntax_check['success']:
                return {
                    'stdout': '',
                    'stderr': syntax_check['stderr'],
                    'exit_code': syntax_check['exit_code'],
                    'execution_time': syntax_check['execution_time'],
                    'success': False,
                    'timeout': syntax_check.get('timeout', False)
                }

            # Run the JavaScript code with Node.js
            result = self._run_command(
                ["node", filepath],
                input_data=stdin_input,
                timeout=timeout
            )
            
            return {
                'stdout': result['stdout'],
                'stderr': result['stderr'],
                'exit_code': result['exit_code'],
                'execution_time': result['execution_time'],
                'success': result['success'],
                'timeout': result.get('timeout', False)
            }
        
        finally:
            self._cleanup_temp_file(filepath)

    def _normalize_javascript_code(self, code):
        """
        Normalize JavaScript source to avoid hidden-token parse failures
        (e.g., HTML entities, zero-width chars, full-width symbols).
        """
        if not code:
            return code

        normalized = html.unescape(code)
        normalized = unicodedata.normalize('NFKC', normalized)

        # Remove hidden characters that can break tokenization.
        normalized = re.sub(r'[\u200B-\u200D\u2060\uFEFF]', '', normalized)
        normalized = normalized.replace('\u00A0', ' ')

        return normalized
    
    def check_syntax(self, code):
        """Check JavaScript syntax without executing"""
        filepath = self._create_temp_file(code)
        
        try:
            # Use node --check to verify syntax
            result = self._run_command(
                ["node", "--check", filepath],
                timeout=5
            )
            
            return {
                'success': result['success'],
                'message': 'Syntax is valid' if result['success'] else result['stderr']
            }
        
        finally:
            self._cleanup_temp_file(filepath)
    
    def analyze(self, code):
        """Analyze JavaScript code structure"""
        base_analysis = super().analyze(code)
        
        # Extract function declarations
        function_declarations = re.findall(r'function\s+(\w+)\s*\(', code)
        
        # Extract arrow functions
        arrow_functions = re.findall(r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>', code)
        
        # Extract classes
        classes = re.findall(r'class\s+(\w+)', code)
        
        # Extract variables
        const_vars = re.findall(r'const\s+(\w+)', code)
        let_vars = re.findall(r'let\s+(\w+)', code)
        var_vars = re.findall(r'var\s+(\w+)', code)
        
        # Extract imports (ES6)
        imports = re.findall(r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]', code)
        
        # Extract requires (CommonJS)
        requires = re.findall(r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)', code)
        
        all_functions = function_declarations + arrow_functions
        
        base_analysis.update({
            'functions': all_functions,
            'classes': classes,
            'constants': const_vars,
            'variables': let_vars + var_vars,
            'imports': imports + requires,
            'function_count': len(all_functions),
            'class_count': len(classes)
        })
        
        return base_analysis
    
    def get_template(self):
        """Get JavaScript code template"""
        return '''// JavaScript Example
// Author: CodeBridge

const readline = require('readline');

function greet(name) {
    return `Hello, ${name}!`;
}

async function main() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });
    
    rl.question('Enter your name: ', (name) => {
        const message = greet(name);
        console.log(message);
        rl.close();
    });
}

main();
'''
