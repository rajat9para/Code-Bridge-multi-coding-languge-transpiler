"""
CodeBridge - C++ Compiler
"""

import os
import re
import tempfile
import shutil
from .base_compiler import BaseCompiler


class CPPCompiler(BaseCompiler):
    """C++ language compiler and executor"""
    
    def __init__(self):
        super().__init__()
        self.language_name = "cpp"
        self.display_name = "C++"
        self.extension = ".cpp"
        self.version_command = ["g++", "--version"]
    
    def execute(self, code, stdin_input="", timeout=30):
        """Execute C++ code"""
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Write C++ source file
            cpp_file = os.path.join(temp_dir, "main.cpp")
            exe_file = os.path.join(temp_dir, "main.exe" if os.name == 'nt' else "main")
            
            with open(cpp_file, 'w') as f:
                f.write(code)
            
            # Compile C++ code
            compile_result = self._run_command(
                ["g++", "-o", exe_file, cpp_file, "-std=c++17"],
                timeout=timeout,
                cwd=temp_dir
            )
            
            if not compile_result['success']:
                return {
                    'stdout': '',
                    'stderr': compile_result['stderr'],
                    'exit_code': compile_result['exit_code'],
                    'execution_time': compile_result['execution_time'],
                    'success': False,
                    'stage': 'compilation'
                }
            
            # Run the executable
            run_result = self._run_command(
                [exe_file],
                input_data=stdin_input,
                timeout=timeout
            )
            
            return {
                'stdout': run_result['stdout'],
                'stderr': run_result['stderr'],
                'exit_code': run_result['exit_code'],
                'execution_time': run_result['execution_time'],
                'success': run_result['success'],
                'timeout': run_result.get('timeout', False),
                'stage': 'execution'
            }
        
        finally:
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
    
    def compile(self, code):
        """Compile C++ code without running"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            cpp_file = os.path.join(temp_dir, "main.cpp")
            exe_file = os.path.join(temp_dir, "main.exe" if os.name == 'nt' else "main")
            
            with open(cpp_file, 'w') as f:
                f.write(code)
            
            result = self._run_command(
                ["g++", "-o", exe_file, cpp_file, "-std=c++17"],
                timeout=30,
                cwd=temp_dir
            )
            
            return {
                'success': result['success'],
                'message': 'Compilation successful' if result['success'] else 'Compilation failed',
                'errors': result['stderr'] if result['stderr'] else None
            }
        
        finally:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
    
    def analyze(self, code):
        """Analyze C++ code structure"""
        base_analysis = super().analyze(code)
        
        # Extract includes
        includes = re.findall(r'#include\s*[<"]([^>"]+)[>"]', code)
        
        # Extract functions (simplified pattern)
        functions = re.findall(r'\b(\w+)\s+(\w+)\s*\([^)]*\)\s*\{', code)
        
        # Extract classes
        classes = re.findall(r'class\s+(\w+)', code)
        
        # Extract namespaces
        namespaces = re.findall(r'namespace\s+(\w+)', code)
        
        base_analysis.update({
            'includes': includes,
            'functions': [f[1] for f in functions if f[1] not in ['if', 'while', 'for', 'switch']],
            'classes': classes,
            'namespaces': namespaces,
            'include_count': len(includes),
            'class_count': len(classes)
        })
        
        return base_analysis
    
    def get_template(self):
        """Get C++ code template"""
        return '''// C++ Example
// Author: CodeBridge

#include <iostream>
#include <string>

std::string greet(const std::string& name) {
    return "Hello, " + name + "!";
}

int main() {
    std::string name;
    std::cout << "Enter your name: ";
    std::getline(std::cin, name);
    std::string message = greet(name);
    std::cout << message << std::endl;
    return 0;
}
'''