"""
CodeBridge - Base Compiler Class
"""

import os
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod


class BaseCompiler(ABC):
    """Abstract base class for all language compilers"""
    
    def __init__(self):
        self.language_name = "base"
        self.display_name = "Base"
        self.extension = ".txt"
        self.version_command = None
        self.compiler_path = None
    
    @abstractmethod
    def execute(self, code, stdin_input="", timeout=30):
        """
        Execute the given code
        
        Args:
            code (str): Source code to execute
            stdin_input (str): Input to pass to stdin
            timeout (int): Maximum execution time in seconds
        
        Returns:
            dict: Execution result with stdout, stderr, exit_code, execution_time
        """
        pass
    
    def get_display_name(self):
        """Get the display name of the language"""
        return self.display_name
    
    def get_extension(self):
        """Get the file extension for the language"""
        return self.extension
    
    def get_version(self):
        """Get the version of the compiler/interpreter"""
        if self.version_command:
            try:
                result = subprocess.run(
                    self.version_command,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return result.stdout.strip().split('\n')[0]
                return "Not installed"
            except Exception:
                return "Not installed"
        return "N/A"
    
    def get_template(self):
        """Get a basic code template for the language"""
        return ""
    
    def check_syntax(self, code):
        """Check syntax of the code without executing"""
        return {
            'success': True,
            'message': 'Syntax check not implemented for this language'
        }
    
    def analyze(self, code):
        """Analyze the code structure"""
        lines = code.split('\n')
        return {
            'lines': len(lines),
            'characters': len(code),
            'non_empty_lines': len([l for l in lines if l.strip()]),
            'language': self.language_name
        }
    
    def _run_command(self, command, input_data=None, timeout=30, cwd=None):
        """
        Run a command with timeout
        
        Args:
            command (list): Command to run
            input_data (str): Data to pass to stdin
            timeout (int): Timeout in seconds
            cwd (str): Working directory
        
        Returns:
            dict: Result with stdout, stderr, exit_code, execution_time
        """
        start_time = time.time()
        
        try:
            process = subprocess.run(
                command,
                input=input_data,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd
            )
            
            execution_time = time.time() - start_time
            
            return {
                'stdout': process.stdout,
                'stderr': process.stderr,
                'exit_code': process.returncode,
                'execution_time': round(execution_time, 3),
                'success': process.returncode == 0
            }
        
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            return {
                'stdout': '',
                'stderr': f'Execution timed out after {timeout} seconds',
                'exit_code': -1,
                'execution_time': round(execution_time, 3),
                'success': False,
                'timeout': True
            }
        
        except FileNotFoundError as e:
            execution_time = time.time() - start_time
            return {
                'stdout': '',
                'stderr': f'Compiler/Interpreter not found: {str(e)}. Please ensure {self.display_name} is installed and in PATH.',
                'exit_code': -1,
                'execution_time': round(execution_time, 3),
                'success': False
            }
        
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'stdout': '',
                'stderr': str(e),
                'exit_code': -1,
                'execution_time': round(execution_time, 3),
                'success': False
            }
    
    def _create_temp_file(self, code, suffix=None):
        """Create a temporary file with the given code"""
        if suffix is None:
            suffix = self.extension
        
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix=suffix,
            delete=False
        )
        temp_file.write(code)
        temp_file.close()
        return temp_file.name
    
    def _cleanup_temp_file(self, filepath):
        """Remove a temporary file"""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass