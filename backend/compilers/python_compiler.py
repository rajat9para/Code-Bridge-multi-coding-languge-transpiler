"""
CodeBridge - Python Compiler
"""

import os
import sys
import ast
from .base_compiler import BaseCompiler


class PythonCompiler(BaseCompiler):
    """Python language compiler and executor"""
    
    def __init__(self):
        super().__init__()
        self.language_name = "python"
        self.display_name = "Python"
        self.extension = ".py"
        self.version_command = [sys.executable, "--version"]
    
    def execute(self, code, stdin_input="", timeout=30):
        """Execute Python code"""
        # Create a temporary file
        filepath = self._create_temp_file(code)
        
        try:
            # Run the Python code
            result = self._run_command(
                [sys.executable, filepath],
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
    
    def check_syntax(self, code):
        """Check Python syntax without executing"""
        try:
            ast.parse(code)
            return {
                'success': True,
                'message': 'Syntax is valid'
            }
        except SyntaxError as e:
            return {
                'success': False,
                'message': f'Syntax error at line {e.lineno}: {e.msg}',
                'line': e.lineno,
                'column': e.offset
            }
    
    def analyze(self, code):
        """Analyze Python code structure"""
        base_analysis = super().analyze(code)
        
        try:
            tree = ast.parse(code)
            
            # Count different node types
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        'name': node.name,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args]
                    })
                elif isinstance(node, ast.ClassDef):
                    classes.append({
                        'name': node.name,
                        'line': node.lineno
                    })
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append(f'{module}.{alias.name}' if module else alias.name)
            
            base_analysis.update({
                'functions': functions,
                'classes': classes,
                'imports': list(set(imports)),
                'function_count': len(functions),
                'class_count': len(classes),
                'import_count': len(set(imports))
            })
        
        except SyntaxError:
            base_analysis['syntax_error'] = True
        
        return base_analysis
    
    def get_template(self):
        """Get Python code template"""
        return '''# Python Example
# Author: CodeBridge

def greet(name):
    """Greet someone by name"""
    return f"Hello, {name}!"

def main():
    name = input("Enter your name: ")
    message = greet(name)
    print(message)

if __name__ == "__main__":
    main()
'''