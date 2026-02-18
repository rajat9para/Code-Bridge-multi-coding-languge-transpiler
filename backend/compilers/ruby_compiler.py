"""
CodeBridge - Ruby Compiler
"""

import os
import re
from .base_compiler import BaseCompiler


class RubyCompiler(BaseCompiler):
    """Ruby language compiler and executor"""
    
    def __init__(self):
        super().__init__()
        self.language_name = "ruby"
        self.display_name = "Ruby"
        self.extension = ".rb"
        self.version_command = ["ruby", "--version"]
    
    def execute(self, code, stdin_input="", timeout=30):
        """Execute Ruby code"""
        # Create a temporary file
        filepath = self._create_temp_file(code)
        
        try:
            # Run the Ruby code
            result = self._run_command(
                ["ruby", filepath],
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
        """Check Ruby syntax without executing"""
        filepath = self._create_temp_file(code)
        
        try:
            # Use ruby -c to check syntax
            result = self._run_command(
                ["ruby", "-c", filepath],
                timeout=5
            )
            
            return {
                'success': result['success'],
                'message': 'Syntax is valid' if result['success'] else result['stderr']
            }
        
        finally:
            self._cleanup_temp_file(filepath)
    
    def analyze(self, code):
        """Analyze Ruby code structure"""
        base_analysis = super().analyze(code)
        
        # Extract methods
        methods = re.findall(r'def\s+(\w+)', code)
        
        # Extract classes
        classes = re.findall(r'class\s+(\w+)', code)
        
        # Extract modules
        modules = re.findall(r'module\s+(\w+)', code)
        
        # Extract requires
        requires = re.findall(r'require\s+[\'"]([^\'"]+)[\'"]', code)
        
        # Extract instance variables
        instance_vars = re.findall(r'@(\w+)', code)
        
        # Extract global variables
        global_vars = re.findall(r'\$(\w+)', code)
        
        base_analysis.update({
            'methods': methods,
            'classes': classes,
            'modules': modules,
            'requires': list(set(requires)),
            'instance_variables': list(set(instance_vars)),
            'global_variables': list(set(global_vars)),
            'method_count': len(methods),
            'class_count': len(classes)
        })
        
        return base_analysis
    
    def get_template(self):
        """Get Ruby code template"""
        return '''# Ruby Example
# Author: CodeBridge

def greet(name)
  "Hello, #{name}!"
end

def main
  print "Enter your name: "
  name = gets.chomp
  message = greet(name)
  puts message
end

main
'''