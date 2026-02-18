"""
CodeBridge - Java Compiler
"""

import os
import re
from .base_compiler import BaseCompiler


class JavaCompiler(BaseCompiler):
    """Java language compiler and executor"""
    
    def __init__(self):
        super().__init__()
        self.language_name = "java"
        self.display_name = "Java"
        self.extension = ".java"
        self.version_command = ["java", "--version"]
    
    def _extract_class_name(self, code):
        """Extract the main class name from Java code"""
        # Look for public class
        match = re.search(r'public\s+class\s+(\w+)', code)
        if match:
            return match.group(1)
        
        # Look for any class
        match = re.search(r'class\s+(\w+)', code)
        if match:
            return match.group(1)
        
        return "Main"
    
    def execute(self, code, stdin_input="", timeout=30):
        """Execute Java code"""
        # Extract class name and create file
        class_name = self._extract_class_name(code)
        
        # Create temp directory for Java files
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Write Java source file
            java_file = os.path.join(temp_dir, f"{class_name}.java")
            with open(java_file, 'w') as f:
                f.write(code)
            
            # Compile Java code
            compile_result = self._run_command(
                ["javac", java_file],
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
            
            # Run Java program
            run_result = self._run_command(
                ["java", class_name],
                input_data=stdin_input,
                timeout=timeout,
                cwd=temp_dir
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
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
    
    def compile(self, code):
        """Compile Java code without running"""
        class_name = self._extract_class_name(code)
        
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        try:
            java_file = os.path.join(temp_dir, f"{class_name}.java")
            with open(java_file, 'w') as f:
                f.write(code)
            
            result = self._run_command(
                ["javac", java_file],
                timeout=30,
                cwd=temp_dir
            )
            
            return {
                'success': result['success'],
                'message': 'Compilation successful' if result['success'] else 'Compilation failed',
                'errors': result['stderr'] if result['stderr'] else None
            }
        
        finally:
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
    
    def analyze(self, code):
        """Analyze Java code structure"""
        base_analysis = super().analyze(code)
        
        # Extract classes
        classes = re.findall(r'class\s+(\w+)', code)
        
        # Extract methods
        methods = re.findall(r'(public|private|protected)?\s*(static)?\s*\w+\s+(\w+)\s*\([^)]*\)', code)
        
        # Extract imports
        imports = re.findall(r'import\s+([\w.]+)', code)
        
        # Extract package
        package_match = re.search(r'package\s+([\w.]+)', code)
        
        base_analysis.update({
            'classes': classes,
            'methods': [m[2] for m in methods],
            'imports': imports,
            'package': package_match.group(1) if package_match else None,
            'class_count': len(classes),
            'method_count': len(methods)
        })
        
        return base_analysis
    
    def get_template(self):
        """Get Java code template"""
        return '''// Java Example
// Author: CodeBridge

import java.util.Scanner;

public class Main {
    public static String greet(String name) {
        return "Hello, " + name + "!";
    }
    
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter your name: ");
        String name = scanner.nextLine();
        String message = greet(name);
        System.out.println(message);
        scanner.close();
    }
}
'''