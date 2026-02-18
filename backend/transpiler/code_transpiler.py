"""
CodeBridge - Enhanced Code Transpiler
Supports complex code structures: loops, conditions, functions, arrays
"""

import ast
import re
from typing import Dict, List, Optional, Tuple


class CodeTranspiler:
    """Enhanced transpiler for Python, Java, C++, JavaScript, Ruby"""
    
    def __init__(self):
        self.languages = ['python', 'java', 'cpp', 'javascript', 'ruby']
        self._reset_context()
    
    def transpile(self, code: str, source_lang: str, target_lang: str) -> Dict:
        """Convert code from source to target language"""
        if source_lang not in self.languages:
            return {'error': f'Unsupported source: {source_lang}'}
        if target_lang not in self.languages:
            return {'error': f'Unsupported target: {target_lang}'}
        
        if source_lang == target_lang:
            return {'original_code': code, 'converted_code': code, 
                    'source_language': source_lang, 'target_language': target_lang}
        
        self._reset_context()

        if source_lang == 'python' and target_lang in ['java', 'cpp', 'javascript', 'ruby']:
            converted = self._convert_python_ast(code, target_lang)
        else:
            converted = self._convert_code(code, source_lang, target_lang)

        converted = self._inject_required_dependencies(converted, target_lang)
        converted = self._ensure_executable_wrapper(converted, target_lang)
        
        return {
            'original_code': code,
            'converted_code': converted,
            'source_language': source_lang,
            'target_language': target_lang
        }

    def _reset_context(self) -> None:
        self._java_declared_stack: List[set] = [set()]
        self._java_type_overrides_stack: List[Dict[str, Optional[str]]] = [{}]
        self._java_var_types: Dict[str, str] = {}
        self._java_arrays: set = set()
        self._java_maps: set = set()
        self._java_imports: set = set()
        self._java_temp_counter = 0
        self._java_function_return_types: Dict[str, str] = {}
        self._java_function_param_types: Dict[str, List[str]] = {}
        self._java_needs_array_concat_helper = False
        self._java_needs_slice_helper = False

        self._cpp_declared_stack: List[set] = [set()]
        self._cpp_type_overrides_stack: List[Dict[str, Optional[str]]] = [{}]
        self._cpp_var_types: Dict[str, str] = {}
        self._cpp_vectors: set = set()
        self._cpp_maps: set = set()
        self._cpp_includes: set = {'iostream'}
        self._cpp_temp_counter = 0
        self._cpp_needs_vector_to_string = False
        self._cpp_function_return_types: Dict[str, str] = {}
        self._cpp_function_param_types: Dict[str, List[str]] = {}
        self._cpp_function_param_mutable: Dict[str, List[bool]] = {}
        self._cpp_needs_vector_concat_helper = False
        self._cpp_needs_slice_helper = False

        self._js_declared_stack: List[set] = [set()]
        self._js_temp_counter = 0
        self._js_var_types: Dict[str, str] = {}
        self._js_function_return_types: Dict[str, str] = {}

    def _inject_required_dependencies(self, code: str, target_lang: str) -> str:
        """Inject required headers/imports based on converted code usage."""
        if not code.strip():
            return code

        if target_lang == 'cpp':
            return self._inject_cpp_headers(code)
        if target_lang == 'java':
            return self._inject_java_imports(code)
        if target_lang == 'python':
            return self._inject_python_imports(code)
        return code

    def _inject_cpp_headers(self, code: str) -> str:
        lines = code.split('\n')
        existing_includes = set()
        for line in lines:
            match = re.match(r'^\s*#include\s*<([^>]+)>', line)
            if match:
                existing_includes.add(match.group(1))

        header_rules = [
            ('iostream', r'(?<!std::)\b(?:cout|cin|endl)\b|\bstd::(?:cout|cin|endl)\b'),
            ('vector', r'\bvector\s*<|\bstd::vector\s*<'),
            ('string', r'\bstring\b|\bstd::string\b'),
            ('algorithm', r'\b(?:sort|reverse|min|max)\s*\('),
            ('cmath', r'\b(?:sqrt|pow|sin|cos|tan|log|exp|fabs|ceil|floor)\s*\('),
            ('map', r'\bmap\s*<|\bstd::map\s*<'),
            ('unordered_map', r'\bunordered_map\s*<|\bstd::unordered_map\s*<'),
            ('set', r'\bset\s*<|\bstd::set\s*<'),
            ('unordered_set', r'\bunordered_set\s*<|\bstd::unordered_set\s*<'),
            ('queue', r'\bqueue\s*<|\bstd::queue\s*<'),
            ('stack', r'\bstack\s*<|\bstd::stack\s*<')
        ]

        includes_to_add = []
        for header, pattern in header_rules:
            if re.search(pattern, code) and header not in existing_includes:
                includes_to_add.append(f'#include <{header}>')

        insert_idx = self._find_leading_insert_index(lines)
        include_block_end = insert_idx
        while include_block_end < len(lines) and re.match(r'^\s*#include\s*<[^>]+>', lines[include_block_end]):
            include_block_end += 1

        if includes_to_add:
            lines[include_block_end:include_block_end] = includes_to_add
            include_block_end += len(includes_to_add)

        has_using_namespace = bool(re.search(r'^\s*using\s+namespace\s+std\s*;', '\n'.join(lines), re.MULTILINE))
        uses_unqualified_std = bool(re.search(
            r'(?<!std::)\b(?:cout|cin|endl|vector|string|map|unordered_map|set|unordered_set|queue|stack)\b',
            code
        ))

        if uses_unqualified_std and not has_using_namespace:
            lines.insert(include_block_end, 'using namespace std;')

        return '\n'.join(lines)

    def _inject_java_imports(self, code: str) -> str:
        lines = code.split('\n')
        existing_imports = set()
        for line in lines:
            match = re.match(r'^\s*import\s+([\w\.\*]+)\s*;', line)
            if match:
                existing_imports.add(match.group(1))

        has_java_util_import = any(name == 'java.util.*' or name.startswith('java.util.') for name in existing_imports)
        has_java_io_import = any(name == 'java.io.*' or name.startswith('java.io.') for name in existing_imports)

        import_rules = [
            ('java.util.*', r'\b(?:Scanner|List|ArrayList|LinkedList|Map|HashMap|Set|HashSet|Collections|Arrays|Deque|Queue|Stack)\b', has_java_util_import),
            ('java.io.*', r'\b(?:BufferedReader|InputStreamReader|FileReader|FileWriter|PrintWriter|IOException)\b', has_java_io_import)
        ]

        imports_to_add = []
        for import_name, pattern, already_satisfied in import_rules:
            if re.search(pattern, code) and not already_satisfied:
                imports_to_add.append(f'import {import_name};')

        if not imports_to_add:
            return code

        insert_idx = self._find_leading_insert_index(lines)
        package_match = re.compile(r'^\s*package\s+[\w\.]+\s*;')
        if insert_idx < len(lines) and package_match.match(lines[insert_idx]):
            insert_idx += 1

        while insert_idx < len(lines) and lines[insert_idx].strip() == '':
            insert_idx += 1

        while insert_idx < len(lines) and re.match(r'^\s*import\s+[\w\.\*]+\s*;', lines[insert_idx]):
            insert_idx += 1

        lines[insert_idx:insert_idx] = imports_to_add
        return '\n'.join(lines)

    def _inject_python_imports(self, code: str) -> str:
        lines = code.split('\n')
        full_code = '\n'.join(lines)

        has_import_sys = bool(re.search(r'^\s*(?:import\s+sys|from\s+sys\s+import)\b', full_code, re.MULTILINE))
        has_import_math = bool(re.search(r'^\s*(?:import\s+math|from\s+math\s+import)\b', full_code, re.MULTILINE))
        has_typing_import = bool(re.search(r'^\s*from\s+typing\s+import\b', full_code, re.MULTILINE))
        has_collections_import = bool(re.search(r'^\s*from\s+collections\s+import\b', full_code, re.MULTILINE))

        imports_to_add = []

        if re.search(r'\bsys\.', code) and not has_import_sys:
            imports_to_add.append('import sys')

        if re.search(r'\bmath\.', code) and not has_import_math:
            imports_to_add.append('import math')

        typing_symbols = []
        typing_symbol_patterns = {
            'List': r'\bList\s*\[',
            'Dict': r'\bDict\s*\[',
            'Set': r'\bSet\s*\[',
            'Tuple': r'\bTuple\s*\[',
            'Optional': r'\bOptional\s*\['
        }
        for symbol, pattern in typing_symbol_patterns.items():
            if re.search(pattern, code):
                typing_symbols.append(symbol)
        if typing_symbols and not has_typing_import:
            imports_to_add.append(f'from typing import {", ".join(sorted(typing_symbols))}')

        collection_symbols = []
        collection_symbol_patterns = {
            'deque': r'\bdeque\s*\(',
            'defaultdict': r'\bdefaultdict\s*\(',
            'Counter': r'\bCounter\s*\('
        }
        for symbol, pattern in collection_symbol_patterns.items():
            if re.search(pattern, code):
                collection_symbols.append(symbol)
        if collection_symbols and not has_collections_import:
            imports_to_add.append(f'from collections import {", ".join(sorted(collection_symbols))}')

        if not imports_to_add:
            return code

        insert_idx = 0
        if lines and lines[0].startswith('#!'):
            insert_idx = 1

        if insert_idx < len(lines) and re.match(r'^#.*coding[:=]\s*[-\w.]+', lines[insert_idx]):
            insert_idx += 1

        if insert_idx < len(lines):
            stripped = lines[insert_idx].strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote = '"""' if stripped.startswith('"""') else "'''"
                if stripped.count(quote) >= 2 and len(stripped) > 3:
                    insert_idx += 1
                else:
                    insert_idx += 1
                    while insert_idx < len(lines) and quote not in lines[insert_idx]:
                        insert_idx += 1
                    if insert_idx < len(lines):
                        insert_idx += 1

        while insert_idx < len(lines) and lines[insert_idx].strip() == '':
            insert_idx += 1

        while insert_idx < len(lines) and re.match(r'^\s*(import|from)\s+\w+', lines[insert_idx]):
            insert_idx += 1

        lines[insert_idx:insert_idx] = imports_to_add
        return '\n'.join(lines)

    def _find_leading_insert_index(self, lines: List[str]) -> int:
        """Find insertion point after leading comments and blank lines."""
        idx = 0
        while idx < len(lines):
            stripped = lines[idx].strip()
            if stripped == '':
                idx += 1
                continue
            if stripped.startswith('//') or stripped.startswith('#'):
                idx += 1
                continue
            if stripped.startswith('/*'):
                idx += 1
                while idx < len(lines) and '*/' not in lines[idx]:
                    idx += 1
                if idx < len(lines):
                    idx += 1
                continue
            break
        return idx

    def _ensure_executable_wrapper(self, code: str, target_lang: str) -> str:
        """Ensure converted code is executable for compiled languages."""
        if target_lang == 'java':
            has_class = bool(re.search(r'\bclass\s+\w+', code))
            has_main = bool(re.search(r'public\s+static\s+void\s+main\s*\(', code))
            if has_class and has_main:
                return code
            return self._wrap_java_program(code)

        if target_lang == 'cpp':
            if re.search(r'\bint\s+main\s*\(', code):
                return code
            return self._wrap_cpp_program(code)

        return code

    def _wrap_java_program(self, code: str) -> str:
        lines = code.split('\n')
        import_lines = []
        code_lines = []
        for line in lines:
            if re.match(r'^\s*import\s+[\w\.\*]+\s*;', line):
                import_lines.append(line.strip())
            elif line.strip():
                code_lines.append(line.rstrip())

        method_lines: List[str] = []
        body_lines: List[str] = []
        i = 0
        method_start = re.compile(r'^\s*(?:public|private|protected)?\s*static\s+[\w<>\[\]]+\s+\w+\s*\([^)]*\)\s*\{')
        while i < len(code_lines):
            line = code_lines[i]
            if method_start.match(line):
                brace_balance = line.count('{') - line.count('}')
                method_lines.append(line)
                i += 1
                while i < len(code_lines) and brace_balance > 0:
                    method_lines.append(code_lines[i])
                    brace_balance += code_lines[i].count('{') - code_lines[i].count('}')
                    i += 1
                continue
            body_lines.append(line)
            i += 1

        wrapped = []
        if import_lines:
            wrapped.extend(import_lines)
            wrapped.append('')

        wrapped.append('public class Main {')
        if method_lines:
            for line in method_lines:
                wrapped.append(f'    {line}')
            wrapped.append('')
        wrapped.append('    public static void main(String[] args) {')
        if body_lines:
            for line in body_lines:
                wrapped.append(f'        {line}')
        else:
            wrapped.append('        // No executable statements')
        wrapped.append('    }')
        wrapped.append('}')
        return '\n'.join(wrapped)

    def _wrap_cpp_program(self, code: str) -> str:
        lines = code.split('\n')
        include_lines = []
        using_lines = []
        code_lines = []

        for line in lines:
            if re.match(r'^\s*#include\s*<[^>]+>', line):
                include_lines.append(line.strip())
            elif re.match(r'^\s*using\s+namespace\s+\w+\s*;', line):
                using_lines.append(line.strip())
            elif line.strip():
                code_lines.append(line.rstrip())

        function_lines: List[str] = []
        body_lines: List[str] = []
        i = 0
        keywords = {'if', 'for', 'while', 'switch'}
        function_start = re.compile(r'^\s*[\w:<>&\*\s]+\s+(\w+)\s*\([^;]*\)\s*\{')
        while i < len(code_lines):
            line = code_lines[i]
            match = function_start.match(line)
            if match and match.group(1) not in keywords:
                brace_balance = line.count('{') - line.count('}')
                function_lines.append(line)
                i += 1
                while i < len(code_lines) and brace_balance > 0:
                    function_lines.append(code_lines[i])
                    brace_balance += code_lines[i].count('{') - code_lines[i].count('}')
                    i += 1
                function_lines.append('')
                continue
            body_lines.append(line)
            i += 1

        wrapped = []
        wrapped.extend(include_lines)
        if using_lines:
            wrapped.extend(using_lines)
        if wrapped:
            wrapped.append('')

        if function_lines:
            wrapped.extend(function_lines)

        wrapped.append('int main() {')
        if body_lines:
            for line in body_lines:
                wrapped.append(f'    {line}')
        else:
            wrapped.append('    // No executable statements')
        wrapped.append('    return 0;')
        wrapped.append('}')
        return '\n'.join(wrapped)

    def _convert_python_ast(self, code: str, target_lang: str) -> str:
        """Structured conversion path for Python to compiled targets."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return self._convert_code(code, 'python', target_lang)

        if target_lang == 'java':
            return self._convert_python_ast_to_java(tree)
        if target_lang == 'cpp':
            return self._convert_python_ast_to_cpp(tree)
        if target_lang == 'javascript':
            return self._convert_python_ast_to_javascript(tree)
        if target_lang == 'ruby':
            return self._convert_python_ast_to_ruby(tree)
        return self._convert_code(code, 'python', target_lang)

    def _node_expr_text(self, node: ast.AST) -> str:
        try:
            return ast.unparse(node).strip()
        except Exception:
            return ast.dump(node, include_attributes=False)

    def _is_match_default_pattern(self, pattern: ast.AST) -> bool:
        return isinstance(pattern, ast.MatchAs) and pattern.name is None and pattern.pattern is None

    def _extract_match_pattern_values(self, pattern: ast.AST, expr_renderer) -> Optional[List[str]]:
        if self._is_match_default_pattern(pattern):
            return []

        if isinstance(pattern, ast.MatchValue):
            return [expr_renderer(pattern.value)]

        if isinstance(pattern, ast.MatchSingleton):
            return [expr_renderer(ast.Constant(pattern.value))]

        if isinstance(pattern, ast.MatchOr):
            values: List[str] = []
            for sub in pattern.patterns:
                extracted = self._extract_match_pattern_values(sub, expr_renderer)
                if extracted is None or extracted == []:
                    return None
                values.extend(extracted)
            return values

        return None

    def _int_literal_value(self, node: ast.AST) -> Optional[int]:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            if isinstance(node.operand.value, int):
                return -node.operand.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd) and isinstance(node.operand, ast.Constant):
            if isinstance(node.operand.value, int):
                return node.operand.value
        return None

    # -----------------------------
    # Python AST -> Java
    # -----------------------------

    def _convert_python_ast_to_java(self, tree: ast.Module) -> str:
        method_lines: List[str] = []
        main_lines: List[str] = []

        for stmt in tree.body:
            if isinstance(stmt, ast.FunctionDef):
                method_lines.extend(self._java_emit_function(stmt, 1))
            else:
                main_lines.extend(self._java_emit_statement(stmt, 2))

        output: List[str] = []
        if self._java_imports:
            for import_name in sorted(self._java_imports):
                output.append(f'import {import_name};')
            output.append('')

        output.append('public class Main {')
        if self._java_needs_array_concat_helper:
            output.extend([
                '    private static int[] concatArrays(int[] left, int[] right) {',
                '        int[] merged = Arrays.copyOf(left, left.length + right.length);',
                '        System.arraycopy(right, 0, merged, left.length, right.length);',
                '        return merged;',
                '    }',
                ''
            ])
        if self._java_needs_slice_helper:
            output.extend([
                '    private static int[] sliceWithStep(int[] source, int start, int end, int step) {',
                '        if (step == 0) return new int[0];',
                '        java.util.ArrayList<Integer> out = new java.util.ArrayList<>();',
                '        if (step > 0) {',
                '            int safeStart = Math.max(0, start);',
                '            int safeEnd = Math.min(source.length, end);',
                '            for (int i = safeStart; i < safeEnd; i += step) {',
                '                out.add(source[i]);',
                '            }',
                '        } else {',
                '            int safeStart = Math.min(source.length - 1, start);',
                '            int safeEnd = Math.max(-1, end);',
                '            for (int i = safeStart; i > safeEnd; i += step) {',
                '                out.add(source[i]);',
                '            }',
                '        }',
                '        return out.stream().mapToInt(Integer::intValue).toArray();',
                '    }',
                ''
            ])
        if method_lines:
            output.extend(method_lines)
            output.append('')

        output.append('    public static void main(String[] args) {')
        if main_lines:
            output.extend(main_lines)
        else:
            output.append('        // No executable statements')
        output.append('    }')
        output.append('}')
        return '\n'.join(output)

    def _java_indent(self, level: int) -> str:
        return '    ' * level

    def _java_current_scope(self) -> set:
        return self._java_declared_stack[-1]

    def _java_push_scope(self) -> None:
        self._java_declared_stack.append(set())
        self._java_type_overrides_stack.append({})

    def _java_pop_scope(self) -> None:
        if len(self._java_declared_stack) > 1:
            self._java_declared_stack.pop()
            overrides = self._java_type_overrides_stack.pop()
            for name, previous in overrides.items():
                if previous is None:
                    self._java_var_types.pop(name, None)
                else:
                    self._java_var_types[name] = previous
            self._java_arrays = {name for name, t in self._java_var_types.items() if t.endswith('[]')}
            self._java_maps = {name for name, t in self._java_var_types.items() if 'Map<' in t or 'HashMap<' in t}

    def _java_is_declared(self, name: str) -> bool:
        return any(name in scope for scope in self._java_declared_stack)

    def _java_declare(self, name: str, declared_type: Optional[str] = None) -> None:
        self._java_current_scope().add(name)
        if declared_type:
            if name not in self._java_type_overrides_stack[-1]:
                self._java_type_overrides_stack[-1][name] = self._java_var_types.get(name)
            self._java_var_types[name] = declared_type
            if declared_type.endswith('[]'):
                self._java_arrays.add(name)
            if 'Map<' in declared_type or 'HashMap<' in declared_type:
                self._java_maps.add(name)

    def _java_merge_types(self, types: List[Optional[str]]) -> Optional[str]:
        known = [t for t in types if t and t != 'var']
        if not known:
            return None
        if any(t.endswith('[]') for t in known):
            array_types = [t for t in known if t.endswith('[]')]
            if len(set(array_types)) == 1:
                return array_types[0]
            return 'int[]'
        if 'String' in known:
            return 'String'
        if 'double' in known:
            return 'double'
        if 'boolean' in known:
            return 'boolean'
        if 'int' in known:
            return 'int'
        return known[0]

    def _java_infer_assigned_name_type(self, func_node: ast.FunctionDef, name: str) -> Optional[str]:
        inferred: List[Optional[str]] = []
        for sub in ast.walk(func_node):
            if isinstance(sub, ast.Assign):
                for target in sub.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        inferred.append(self._java_infer_type(sub.value))
            elif isinstance(sub, ast.AnnAssign):
                if isinstance(sub.target, ast.Name) and sub.target.id == name and sub.value is not None:
                    inferred.append(self._java_infer_type(sub.value))
        return self._java_merge_types(inferred)

    def _java_infer_param_type(self, func_node: ast.FunctionDef, param_name: str) -> str:
        for sub in ast.walk(func_node):
            if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name) and sub.value.id == param_name:
                return 'int[]'
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == 'len':
                if sub.args and isinstance(sub.args[0], ast.Name) and sub.args[0].id == param_name:
                    return 'int[]'
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in ['append', 'pop']:
                if sub.args and isinstance(sub.args[0], ast.Name) and sub.args[0].id == param_name:
                    return 'int[]'
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                callee = sub.func.id
                if callee in self._java_function_param_types:
                    for arg_idx, arg in enumerate(sub.args):
                        if isinstance(arg, ast.Name) and arg.id == param_name and arg_idx < len(self._java_function_param_types[callee]):
                            expected = self._java_function_param_types[callee][arg_idx]
                            if expected.endswith('[]'):
                                return expected
        return 'int'

    def _java_boxed_type(self, primitive_type: str) -> str:
        mapping = {
            'int': 'Integer',
            'double': 'Double',
            'boolean': 'Boolean',
            'char': 'Character'
        }
        return mapping.get(primitive_type, primitive_type)

    def _java_detect_array_append(self, name: str, value: ast.AST) -> Optional[ast.AST]:
        if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Add):
            return None

        if isinstance(value.left, ast.Name) and value.left.id == name and isinstance(value.right, ast.List) and len(value.right.elts) == 1:
            return value.right.elts[0]
        if isinstance(value.right, ast.Name) and value.right.id == name and isinstance(value.left, ast.List) and len(value.left.elts) == 1:
            return value.left.elts[0]
        return None

    def _java_emit_dict_assign(self, name: str, dict_node: ast.Dict, indent_level: int) -> List[str]:
        indent = self._java_indent(indent_level)
        lines: List[str] = []
        self._java_imports.add('java.util.HashMap')

        key_type = 'String'
        value_type = 'Integer'
        if dict_node.values:
            first_value_type = self._java_infer_type(dict_node.values[0]) or 'int'
            if first_value_type == 'double':
                value_type = 'Double'
            elif first_value_type == 'boolean':
                value_type = 'Boolean'
            elif first_value_type == 'String':
                value_type = 'String'

        map_type = f'HashMap<{key_type}, {value_type}>'
        if self._java_is_declared(name):
            lines.append(f'{indent}{name}.clear();')
        else:
            self._java_declare(name, map_type)
            lines.append(f'{indent}{map_type} {name} = new HashMap<>();')

        for key, value in zip(dict_node.keys, dict_node.values):
            lines.append(f'{indent}{name}.put({self._java_expr(key)}, {self._java_expr(value)});')
        return lines

    def _java_emit_list_comp_assign(self, name: str, comp_node: ast.ListComp, indent_level: int) -> List[str]:
        indent = self._java_indent(indent_level)
        if len(comp_node.generators) != 1:
            return [f'{indent}// Unsupported list comprehension']

        generator = comp_node.generators[0]
        if not isinstance(generator.target, ast.Name):
            return [f'{indent}// Unsupported list comprehension target']

        item_name = generator.target.id
        item_type = self._java_infer_type(comp_node.elt) or 'int'
        item_type = 'int' if item_type == 'var' else item_type
        boxed_type = self._java_boxed_type(item_type)

        temp_id = self._java_temp_counter
        self._java_temp_counter += 1
        list_name = f'__list{temp_id}'
        lines = [f'{indent}ArrayList<{boxed_type}> {list_name} = new ArrayList<>();']
        self._java_imports.add('java.util.ArrayList')

        iter_node = generator.iter
        loop_indent = indent
        body_indent = self._java_indent(indent_level + 1)

        if isinstance(iter_node, ast.Subscript) and isinstance(iter_node.slice, ast.Slice):
            base_expr = self._java_expr(iter_node.value)
            base_type = self._java_infer_type(iter_node.value) or 'int[]'
            if base_type.endswith('[]'):
                start_expr = self._java_expr(iter_node.slice.lower) if iter_node.slice.lower else '0'
                stop_expr = self._java_expr(iter_node.slice.upper) if iter_node.slice.upper else f'{base_expr}.length'
                idx_name = f'__idx{temp_id}'
                lines.append(f'{loop_indent}for (int {idx_name} = {start_expr}; {idx_name} < {stop_expr}; {idx_name}++) {{')
                lines.append(f'{body_indent}int {item_name} = {base_expr}[{idx_name}];')
            else:
                lines.append(f'{loop_indent}for (var {item_name} : {self._java_expr(iter_node)}) {{')
        else:
            iter_expr = self._java_expr(iter_node)
            iter_type = self._java_infer_type(iter_node)
            loop_type = 'int' if iter_type == 'int[]' else 'var'
            lines.append(f'{loop_indent}for ({loop_type} {item_name} : {iter_expr}) {{')

        if generator.ifs:
            cond_expr = ' && '.join(self._java_expr(cond) for cond in generator.ifs)
            lines.append(f'{body_indent}if ({cond_expr}) {{')
            lines.append(f'{self._java_indent(indent_level + 2)}{list_name}.add({self._java_expr(comp_node.elt)});')
            lines.append(f'{body_indent}}}')
        else:
            lines.append(f'{body_indent}{list_name}.add({self._java_expr(comp_node.elt)});')
        lines.append(f'{loop_indent}}}')

        if item_type == 'int':
            self._java_imports.add('java.util.Arrays')
            assign_expr = f'{list_name}.stream().mapToInt(Integer::intValue).toArray()'
            declared_type = 'int[]'
        elif item_type == 'double':
            assign_expr = f'{list_name}.stream().mapToDouble(Double::doubleValue).toArray()'
            declared_type = 'double[]'
        else:
            assign_expr = f'{list_name}.toArray(new {item_type}[0])'
            declared_type = f'{item_type}[]'

        if self._java_is_declared(name):
            lines.append(f'{indent}{name} = {assign_expr};')
        else:
            self._java_declare(name, declared_type)
            lines.append(f'{indent}{declared_type} {name} = {assign_expr};')
        return lines

    def _java_emit_function(self, node: ast.FunctionDef, indent_level: int) -> List[str]:
        self._java_push_scope()

        param_types: Dict[str, str] = {}
        for arg in node.args.args:
            param_types[arg.arg] = self._java_infer_param_type(node, arg.arg)

        params = []
        for arg in node.args.args:
            param_type = param_types[arg.arg]
            params.append(f'{param_type} {arg.arg}')
            self._java_declare(arg.arg, param_type)
        self._java_function_param_types[node.name] = [param_types[arg.arg] for arg in node.args.args]

        return_nodes = [n for n in ast.walk(node) if isinstance(n, ast.Return) and n.value is not None]
        inferred_returns: List[Optional[str]] = []
        for ret in return_nodes:
            inferred = self._java_infer_type(ret.value)
            if inferred is None and isinstance(ret.value, ast.Name):
                inferred = self._java_infer_assigned_name_type(node, ret.value.id)
            inferred_returns.append(inferred)
        return_type = self._java_merge_types(inferred_returns) if inferred_returns else 'void'
        return_type = return_type or 'void'
        self._java_function_return_types[node.name] = return_type

        lines = [f'{self._java_indent(indent_level)}public static {return_type} {node.name}({", ".join(params)}) {{']
        body_lines: List[str] = []
        for stmt in node.body:
            body_lines.extend(self._java_emit_statement(stmt, indent_level + 1))
        if not body_lines:
            body_lines.append(f'{self._java_indent(indent_level + 1)}// No-op')
        lines.extend(body_lines)

        has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
        if return_type != 'void' and not has_return:
            lines.append(f'{self._java_indent(indent_level + 1)}return {self._java_default_return(return_type)};')

        lines.append(f'{self._java_indent(indent_level)}}}')
        self._java_pop_scope()
        return lines

    def _java_emit_statement(self, stmt: ast.stmt, indent_level: int) -> List[str]:
        indent = self._java_indent(indent_level)

        if isinstance(stmt, ast.Assign):
            return self._java_emit_assign(stmt, indent_level)

        if isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.value is not None:
                assign_node = ast.Assign(targets=[stmt.target], value=stmt.value)
                return self._java_emit_assign(assign_node, indent_level)
            return [f'{indent}// Unsupported annotated assignment']

        if isinstance(stmt, ast.AugAssign):
            target_expr = self._java_expr(stmt.target)
            value_expr = self._java_expr(stmt.value)
            op = self._java_binop(stmt.op)
            return [f'{indent}{target_expr} {op}= {value_expr};']

        if isinstance(stmt, ast.Expr):
            if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == 'print':
                return self._java_emit_print(stmt.value, indent_level)
            if isinstance(stmt.value, ast.Call):
                method_lines = self._java_emit_collection_call(stmt.value, indent_level)
                if method_lines is not None:
                    return method_lines
            return [f'{indent}{self._java_expr(stmt.value)};']

        if isinstance(stmt, ast.For):
            return self._java_emit_for(stmt, indent_level)

        if isinstance(stmt, ast.While):
            condition = self._java_expr(stmt.test)
            lines = [f'{indent}while ({condition}) {{']
            for inner in stmt.body:
                lines.extend(self._java_emit_statement(inner, indent_level + 1))
            lines.append(f'{indent}}}')
            return lines

        if isinstance(stmt, ast.If):
            return self._java_emit_if(stmt, indent_level)

        if isinstance(stmt, ast.Match):
            return self._java_emit_match(stmt, indent_level)

        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                return [f'{indent}return;']
            return [f'{indent}return {self._java_expr(stmt.value)};']

        if isinstance(stmt, ast.Break):
            return [f'{indent}break;']

        if isinstance(stmt, ast.Continue):
            return [f'{indent}continue;']

        if isinstance(stmt, ast.Pass):
            return []

        return [f'{indent}// Unsupported statement: {stmt.__class__.__name__}']

    def _java_emit_collection_call(self, call_node: ast.Call, indent_level: int) -> Optional[List[str]]:
        if not isinstance(call_node.func, ast.Attribute):
            return None
        if not isinstance(call_node.func.value, ast.Name):
            return None

        target_name = call_node.func.value.id
        method = call_node.func.attr
        target_type = self._java_var_types.get(target_name, '')
        indent = self._java_indent(indent_level)

        if method == 'append' and len(call_node.args) == 1:
            arg_expr = self._java_expr(call_node.args[0])
            if target_type.endswith('[]') or target_name in self._java_arrays:
                self._java_imports.add('java.util.Arrays')
                elem_type = target_type[:-2] if target_type.endswith('[]') else 'int'
                temp_name = f'__arr{self._java_temp_counter}'
                self._java_temp_counter += 1
                return [
                    f'{indent}{elem_type}[] {temp_name} = Arrays.copyOf({target_name}, {target_name}.length + 1);',
                    f'{indent}{temp_name}[{temp_name}.length - 1] = {arg_expr};',
                    f'{indent}{target_name} = {temp_name};'
                ]
            return [f'{indent}{target_name}.add({arg_expr});']

        if method == 'extend' and len(call_node.args) == 1:
            arg_expr = self._java_expr(call_node.args[0])
            if target_type.endswith('[]') or target_name in self._java_arrays:
                self._java_needs_array_concat_helper = True
                self._java_imports.add('java.util.Arrays')
                return [f'{indent}{target_name} = concatArrays({target_name}, {arg_expr});']
            return [f'{indent}{target_name}.addAll({arg_expr});']

        if method == 'pop' and len(call_node.args) == 0:
            if target_type.endswith('[]') or target_name in self._java_arrays:
                self._java_imports.add('java.util.Arrays')
                return [
                    f'{indent}if ({target_name}.length > 0) {{',
                    f'{self._java_indent(indent_level + 1)}{target_name} = Arrays.copyOf({target_name}, {target_name}.length - 1);',
                    f'{indent}}}'
                ]
            return [f'{indent}if (!{target_name}.isEmpty()) {{ {target_name}.remove({target_name}.size() - 1); }}']

        return None

    def _java_emit_assign(self, stmt: ast.Assign, indent_level: int) -> List[str]:
        indent = self._java_indent(indent_level)

        if len(stmt.targets) != 1:
            if all(isinstance(target, ast.Name) for target in stmt.targets):
                value_expr = self._java_expr(stmt.value)
                inferred_type = self._java_infer_type(stmt.value) or 'var'
                lines: List[str] = []
                for target in stmt.targets:
                    name = target.id
                    if self._java_is_declared(name):
                        lines.append(f'{indent}{name} = {value_expr};')
                    else:
                        self._java_declare(name, inferred_type)
                        lines.append(f'{indent}{inferred_type} {name} = {value_expr};')
                return lines
            return [f'{indent}// Unsupported multiple-target assignment']

        target = stmt.targets[0]
        value = stmt.value

        if (
            isinstance(target, ast.Tuple)
            and isinstance(value, ast.Tuple)
            and len(target.elts) == 2
            and len(value.elts) == 2
            and self._node_expr_text(value.elts[0]) == self._node_expr_text(target.elts[1])
            and self._node_expr_text(value.elts[1]) == self._node_expr_text(target.elts[0])
        ):
            left_a = self._java_expr(target.elts[0])
            left_b = self._java_expr(target.elts[1])
            temp_name = f'__tmp{self._java_temp_counter}'
            self._java_temp_counter += 1
            temp_type = self._java_infer_type(target.elts[0]) or 'int'
            return [
                f'{indent}{temp_type} {temp_name} = {left_a};',
                f'{indent}{left_a} = {left_b};',
                f'{indent}{left_b} = {temp_name};'
            ]

        if isinstance(target, ast.Name):
            name = target.id

            if isinstance(value, ast.Dict):
                return self._java_emit_dict_assign(name, value, indent_level)

            if isinstance(value, ast.ListComp):
                return self._java_emit_list_comp_assign(name, value, indent_level)

            append_expr = self._java_detect_array_append(name, value)
            if append_expr is not None:
                self._java_imports.add('java.util.Arrays')
                item_expr = self._java_expr(append_expr)
                if not self._java_is_declared(name):
                    self._java_declare(name, 'int[]')
                    return [f'{indent}int[] {name} = new int[]{{{item_expr}}};']
                temp_name = f'__arr{self._java_temp_counter}'
                self._java_temp_counter += 1
                return [
                    f'{indent}int[] {temp_name} = Arrays.copyOf({name}, {name}.length + 1);',
                    f'{indent}{temp_name}[{temp_name}.length - 1] = {item_expr};',
                    f'{indent}{name} = {temp_name};'
                ]

            if isinstance(value, ast.List):
                element_type = self._java_infer_list_element_type(value)
                values = ', '.join(self._java_expr(item) for item in value.elts)
                if self._java_is_declared(name):
                    return [f'{indent}{name} = new {element_type}[]{{{values}}};']
                self._java_declare(name, f'{element_type}[]')
                return [f'{indent}{element_type}[] {name} = {{{values}}};']

            value_expr = self._java_expr(value)
            if self._java_is_declared(name):
                return [f'{indent}{name} = {value_expr};']

            inferred_type = self._java_infer_type(value) or 'var'
            self._java_declare(name, inferred_type)
            return [f'{indent}{inferred_type} {name} = {value_expr};']

        if isinstance(target, ast.Subscript):
            right = self._java_expr(value)
            base_type = self._java_infer_type(target.value)
            if isinstance(target.value, ast.Name) and (
                target.value.id in self._java_maps or (base_type and ('Map<' in base_type or 'HashMap<' in base_type))
            ):
                map_expr = self._java_expr(target.value)
                key_expr = self._java_expr(target.slice)
                return [f'{indent}{map_expr}.put({key_expr}, {right});']

            left = self._java_expr(target)
            return [f'{indent}{left} = {right};']

        return [f'{indent}// Unsupported assignment target']

    def _java_emit_for(self, stmt: ast.For, indent_level: int) -> List[str]:
        indent = self._java_indent(indent_level)
        lines: List[str] = []

        var_name = stmt.target.id if isinstance(stmt.target, ast.Name) else 'item'

        if isinstance(stmt.iter, ast.Call) and isinstance(stmt.iter.func, ast.Name) and stmt.iter.func.id == 'range':
            args = stmt.iter.args
            start_expr = '0'
            stop_expr = '0'
            step_expr = '1'
            step_value = 1

            if len(args) == 1:
                stop_expr = self._java_expr(args[0])
            elif len(args) == 2:
                start_expr = self._java_expr(args[0])
                stop_expr = self._java_expr(args[1])
            elif len(args) >= 3:
                start_expr = self._java_expr(args[0])
                stop_expr = self._java_expr(args[1])
                step_expr = self._java_expr(args[2])
                literal_step = self._int_literal_value(args[2])
                if literal_step is not None:
                    step_value = literal_step

            comparator = '>' if step_value < 0 else '<'
            if step_value == 1:
                update = f'{var_name}++'
            elif step_value == -1:
                update = f'{var_name}--'
            else:
                update = f'{var_name} += {step_expr}'

            lines.append(f'{indent}for (int {var_name} = {start_expr}; {var_name} {comparator} {stop_expr}; {update}) {{')
        else:
            iter_expr = self._java_expr(stmt.iter)
            iter_type = self._java_infer_type(stmt.iter)
            element_type = iter_type[:-2] if iter_type and iter_type.endswith('[]') else 'var'
            lines.append(f'{indent}for ({element_type} {var_name} : {iter_expr}) {{')

        for inner in stmt.body:
            lines.extend(self._java_emit_statement(inner, indent_level + 1))
        lines.append(f'{indent}}}')
        return lines

    def _java_emit_if(self, stmt: ast.If, indent_level: int) -> List[str]:
        indent = self._java_indent(indent_level)
        branches = []
        else_body: List[ast.stmt] = []
        current = stmt

        while True:
            branches.append((current.test, current.body))
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
                continue
            else_body = current.orelse
            break

        lines: List[str] = []
        for idx, (test_node, body_nodes) in enumerate(branches):
            keyword = 'if' if idx == 0 else 'else if'
            lines.append(f'{indent}{keyword} ({self._java_expr(test_node)}) {{')
            for body_stmt in body_nodes:
                lines.extend(self._java_emit_statement(body_stmt, indent_level + 1))
            lines.append(f'{indent}}}')

        if else_body:
            lines.append(f'{indent}else {{')
            for body_stmt in else_body:
                lines.extend(self._java_emit_statement(body_stmt, indent_level + 1))
            lines.append(f'{indent}}}')

        return lines

    def _java_emit_match(self, stmt: ast.Match, indent_level: int) -> List[str]:
        indent = self._java_indent(indent_level)
        subject_expr = self._java_expr(stmt.subject)
        subject_type = self._java_infer_type(stmt.subject) or 'int'
        lines: List[str] = []
        has_branch = False

        for idx, case in enumerate(stmt.cases):
            values = self._extract_match_pattern_values(case.pattern, self._java_expr)
            branch_header = ''
            if values == []:
                branch_header = f'{indent}else {{' if has_branch else f'{indent}{{'
            elif values is not None:
                checks = []
                for value_expr in values:
                    if subject_type == 'String':
                        checks.append(f'{subject_expr}.equals({value_expr})')
                    else:
                        checks.append(f'{subject_expr} == {value_expr}')
                cond = ' || '.join(f'({c})' for c in checks)
                branch_header = f'{indent}if ({cond}) {{' if not has_branch else f'{indent}else if ({cond}) {{'
            else:
                lines.append(f'{indent}// Unsupported match pattern')
                continue

            lines.append(branch_header)
            for body_stmt in case.body:
                lines.extend(self._java_emit_statement(body_stmt, indent_level + 1))
            lines.append(f'{indent}}}')
            has_branch = True

        if not has_branch:
            return [f'{indent}// Unsupported match statement']
        return lines

    def _java_emit_print(self, call_node: ast.Call, indent_level: int) -> List[str]:
        indent = self._java_indent(indent_level)
        parts = [self._java_expr_for_print(arg) for arg in call_node.args]
        if not parts:
            return [f'{indent}System.out.println();']
        if len(parts) == 1:
            return [f'{indent}System.out.println({parts[0]});']
        return [f'{indent}System.out.println({" + \" \" + ".join(parts)});']

    def _java_expr_for_print(self, node: ast.AST) -> str:
        node_type = self._java_infer_type(node)
        if node_type and node_type.endswith('[]'):
            self._java_imports.add('java.util.Arrays')
            return f'Arrays.toString({self._java_expr(node)})'
        if node_type == 'boolean':
            expr = self._java_expr(node)
            return f'(({expr}) ? "True" : "False")'

        if isinstance(node, ast.List):
            self._java_imports.add('java.util.Arrays')
            elem_type = self._java_infer_list_element_type(node)
            values = ', '.join(self._java_expr(item) for item in node.elts)
            return f'Arrays.toString(new {elem_type}[]{{{values}}})'

        return self._java_expr(node)

    def _java_expr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, str):
                escaped = value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                return f'"{escaped}"'
            if value is True:
                return 'true'
            if value is False:
                return 'false'
            if value is None:
                return 'null'
            return str(value)

        if isinstance(node, ast.JoinedStr):
            return self._java_fstring(node)

        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.BinOp):
            left = self._java_expr(node.left)
            right = self._java_expr(node.right)
            if isinstance(node.op, ast.Add):
                left_type = self._java_infer_type(node.left)
                right_type = self._java_infer_type(node.right)
                if (left_type and left_type.endswith('[]')) or (right_type and right_type.endswith('[]')):
                    self._java_needs_array_concat_helper = True
                    self._java_imports.add('java.util.Arrays')
                    if not (left_type and left_type.endswith('[]')):
                        left = f'new int[]{{{left}}}'
                    if not (right_type and right_type.endswith('[]')):
                        right = f'new int[]{{{right}}}'
                    return f'concatArrays({left}, {right})'
            op = self._java_binop(node.op)
            return f'({left} {op} {right})'

        if isinstance(node, ast.BoolOp):
            op = ' && ' if isinstance(node.op, ast.And) else ' || '
            return f'({op.join(self._java_expr(v) for v in node.values)})'

        if isinstance(node, ast.UnaryOp):
            operand = self._java_expr(node.operand)
            if isinstance(node.op, ast.Not):
                return f'(!{operand})'
            if isinstance(node.op, ast.USub):
                return f'(-{operand})'
            return operand

        if isinstance(node, ast.Compare):
            left_expr = self._java_expr(node.left)
            parts = []
            for op, comparator in zip(node.ops, node.comparators):
                right_expr = self._java_expr(comparator)
                parts.append(f'({left_expr} {self._java_compare_op(op)} {right_expr})')
                left_expr = right_expr
            return ' && '.join(parts)

        if isinstance(node, ast.Subscript):
            base_expr = self._java_expr(node.value)
            base_type = self._java_infer_type(node.value)
            if isinstance(node.slice, ast.Slice):
                step_expr = self._java_expr(node.slice.step) if node.slice.step else None
                literal_step = self._int_literal_value(node.slice.step) if node.slice.step else None
                default_start = '0'
                default_end = f'{base_expr}.length'
                if literal_step is not None and literal_step < 0:
                    default_start = f'{base_expr}.length - 1'
                    default_end = '-1'
                start_expr = self._java_expr(node.slice.lower) if node.slice.lower else default_start
                if base_type == 'String':
                    end_expr = self._java_expr(node.slice.upper) if node.slice.upper else f'{base_expr}.length()'
                    return f'{base_expr}.substring({start_expr}, {end_expr})'
                end_expr = self._java_expr(node.slice.upper) if node.slice.upper else default_end
                if step_expr is not None and (literal_step is None or literal_step != 1):
                    self._java_needs_slice_helper = True
                    return f'sliceWithStep({base_expr}, {start_expr}, {end_expr}, {step_expr})'
                self._java_imports.add('java.util.Arrays')
                return f'Arrays.copyOfRange({base_expr}, {start_expr}, {end_expr})'
            index_expr = self._java_expr(node.slice)
            if base_type == 'String':
                return f'{base_expr}.charAt({index_expr})'
            if isinstance(node.value, ast.Name) and node.value.id in self._java_maps:
                return f'{base_expr}.get({index_expr})'
            return f'{base_expr}[{index_expr}]'

        if isinstance(node, ast.Attribute):
            return f'{self._java_expr(node.value)}.{node.attr}'

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name == 'len' and len(node.args) == 1:
                    arg_expr = self._java_expr(node.args[0])
                    arg_type = self._java_infer_type(node.args[0])
                    if arg_type == 'String':
                        return f'{arg_expr}.length()'
                    if arg_type and ('Map<' in arg_type or 'HashMap<' in arg_type):
                        return f'{arg_expr}.size()'
                    return f'{arg_expr}.length'
                if func_name == 'str' and len(node.args) == 1:
                    return f'String.valueOf({self._java_expr(node.args[0])})'
                if func_name == 'int' and len(node.args) == 1:
                    return f'((int) ({self._java_expr(node.args[0])}))'
                if func_name == 'float' and len(node.args) == 1:
                    return f'((double) ({self._java_expr(node.args[0])}))'
                if func_name in self._java_function_return_types:
                    args = ', '.join(self._java_expr(arg) for arg in node.args)
                    return f'{func_name}({args})'
            args = ', '.join(self._java_expr(arg) for arg in node.args)
            return f'{self._java_expr(node.func)}({args})'

        if isinstance(node, ast.List):
            elem_type = self._java_infer_list_element_type(node)
            values = ', '.join(self._java_expr(item) for item in node.elts)
            return f'new {elem_type}[]{{{values}}}'

        if isinstance(node, ast.Tuple):
            return ', '.join(self._java_expr(item) for item in node.elts)

        if isinstance(node, ast.Dict):
            return '/* unsupported dict literal */'

        if isinstance(node, ast.ListComp):
            return '/* unsupported list comprehension */'

        return '/* unsupported */'

    def _java_fstring(self, node: ast.JoinedStr) -> str:
        parts: List[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                escaped = value.value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                parts.append(f'"{escaped}"')
            elif isinstance(value, ast.FormattedValue):
                parts.append(f'String.valueOf({self._java_expr(value.value)})')

        if not parts:
            return '""'
        return ' + '.join(parts)

    def _java_binop(self, op_node: ast.AST) -> str:
        mapping = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.FloorDiv: '/',
            ast.Mod: '%',
            ast.Pow: '*'
        }
        for op_type, symbol in mapping.items():
            if isinstance(op_node, op_type):
                return symbol
        return '+'

    def _java_compare_op(self, op_node: ast.AST) -> str:
        mapping = {
            ast.Eq: '==',
            ast.NotEq: '!=',
            ast.Lt: '<',
            ast.LtE: '<=',
            ast.Gt: '>',
            ast.GtE: '>='
        }
        for op_type, symbol in mapping.items():
            if isinstance(op_node, op_type):
                return symbol
        return '=='

    def _java_infer_list_element_type(self, list_node: ast.List) -> str:
        if not list_node.elts:
            return 'int'
        inferred = [self._java_infer_type(item) for item in list_node.elts]
        if any(t == 'String' for t in inferred):
            return 'String'
        if any(t == 'double' for t in inferred):
            return 'double'
        if any(t == 'boolean' for t in inferred):
            return 'boolean'
        return 'int'

    def _java_infer_type(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool):
                return 'boolean'
            if isinstance(value, int):
                return 'int'
            if isinstance(value, float):
                return 'double'
            if isinstance(value, str):
                return 'String'
            return None

        if isinstance(node, ast.JoinedStr):
            return 'String'

        if isinstance(node, ast.Name):
            return self._java_var_types.get(node.id)

        if isinstance(node, ast.List):
            return f'{self._java_infer_list_element_type(node)}[]'

        if isinstance(node, ast.Dict):
            key_type = 'String'
            value_type = 'Integer'
            if node.values:
                vt = self._java_infer_type(node.values[0]) or 'int'
                if vt == 'double':
                    value_type = 'Double'
                elif vt == 'boolean':
                    value_type = 'Boolean'
                elif vt == 'String':
                    value_type = 'String'
            return f'HashMap<{key_type}, {value_type}>'

        if isinstance(node, ast.ListComp):
            elem_type = self._java_infer_type(node.elt) or 'int'
            elem_type = 'int' if elem_type == 'var' else elem_type
            return f'{elem_type}[]'

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'len':
                return 'int'
            if isinstance(node.func, ast.Name) and node.func.id in self._java_function_return_types:
                return self._java_function_return_types[node.func.id]
            return None

        if isinstance(node, ast.BoolOp) or isinstance(node, ast.Compare):
            return 'boolean'

        if isinstance(node, ast.Subscript):
            parent_type = self._java_infer_type(node.value)
            if isinstance(node.slice, ast.Slice):
                if parent_type == 'String':
                    return 'String'
                if parent_type and parent_type.endswith('[]'):
                    return parent_type
                return None
            if parent_type and ('Map<' in parent_type or 'HashMap<' in parent_type):
                match = re.search(r'<[^,]+,\s*([^>]+)>', parent_type)
                if match:
                    mapped = match.group(1).strip()
                    primitive_map = {'Integer': 'int', 'Double': 'double', 'Boolean': 'boolean'}
                    return primitive_map.get(mapped, mapped)
            if parent_type and parent_type.endswith('[]'):
                return parent_type[:-2]
            if parent_type == 'String':
                return 'char'
            return None

        if isinstance(node, ast.BinOp):
            left_type = self._java_infer_type(node.left)
            right_type = self._java_infer_type(node.right)
            if isinstance(node.op, ast.Add):
                if left_type and left_type.endswith('[]'):
                    return left_type
                if right_type and right_type.endswith('[]'):
                    return right_type
            if left_type == 'double' or right_type == 'double':
                return 'double'
            if left_type == 'String' or right_type == 'String':
                return 'String'
            return 'int'

        return None

    def _java_default_return(self, return_type: str) -> str:
        if return_type == 'int':
            return '0'
        if return_type == 'double':
            return '0.0'
        if return_type == 'boolean':
            return 'false'
        if return_type == 'String':
            return '""'
        if return_type == 'char':
            return "'\\0'"
        if return_type.endswith('[]'):
            base = return_type[:-2]
            return f'new {base}[0]'
        return 'null'

    # -----------------------------
    # Python AST -> C++
    # -----------------------------

    def _convert_python_ast_to_cpp(self, tree: ast.Module) -> str:
        function_blocks: List[List[str]] = []
        function_prototypes: List[str] = []
        main_lines: List[str] = []

        for stmt in tree.body:
            if isinstance(stmt, ast.FunctionDef):
                block = self._cpp_emit_function(stmt, 0)
                if block:
                    function_blocks.append(block)
                    signature = block[0].strip()
                    if signature.endswith('{'):
                        signature = signature[:-1].rstrip()
                    function_prototypes.append(f'{signature};')
            else:
                main_lines.extend(self._cpp_emit_statement(stmt, 1))

        output: List[str] = []

        if self._cpp_needs_slice_helper:
            self._cpp_includes.update({'vector', 'algorithm'})
        if self._cpp_needs_vector_to_string:
            self._cpp_includes.update({'vector', 'string', 'sstream'})
        if self._cpp_needs_vector_concat_helper:
            self._cpp_includes.add('vector')

        include_order = ['iostream', 'vector', 'string', 'sstream', 'algorithm', 'cmath', 'map', 'unordered_map', 'set', 'unordered_set', 'queue', 'stack']
        for header in include_order:
            if header in self._cpp_includes:
                output.append(f'#include <{header}>')
        output.append('using namespace std;')
        output.append('')

        if self._cpp_needs_vector_to_string:
            output.extend([
                'template <typename T>',
                'string vectorToString(const vector<T>& values) {',
                '    ostringstream oss;',
                '    oss << "[";',
                '    for (size_t i = 0; i < values.size(); ++i) {',
                '        if (i > 0) oss << ", ";',
                '        oss << values[i];',
                '    }',
                '    oss << "]";',
                '    return oss.str();',
                '}',
                ''
            ])

        if self._cpp_needs_vector_concat_helper:
            output.extend([
                'template <typename T>',
                'vector<T> concatVectors(const vector<T>& left, const vector<T>& right) {',
                '    vector<T> merged;',
                '    merged.reserve(left.size() + right.size());',
                '    merged.insert(merged.end(), left.begin(), left.end());',
                '    merged.insert(merged.end(), right.begin(), right.end());',
                '    return merged;',
                '}',
                ''
            ])

        if self._cpp_needs_slice_helper:
            output.extend([
                'template <typename T>',
                'vector<T> sliceWithStep(const vector<T>& source, int start, int end, int step) {',
                '    vector<T> out;',
                '    if (step == 0) return out;',
                '    if (step > 0) {',
                '        int safeStart = max(0, start);',
                '        int safeEnd = min(static_cast<int>(source.size()), end);',
                '        for (int i = safeStart; i < safeEnd; i += step) {',
                '            out.push_back(source[i]);',
                '        }',
                '    } else {',
                '        int safeStart = min(static_cast<int>(source.size()) - 1, start);',
                '        int safeEnd = max(-1, end);',
                '        for (int i = safeStart; i > safeEnd; i += step) {',
                '            out.push_back(source[i]);',
                '        }',
                '    }',
                '    return out;',
                '}',
                ''
            ])

        if function_prototypes:
            output.extend(function_prototypes)
            output.append('')

        if function_blocks:
            for block in function_blocks:
                output.extend(block)
                output.append('')

        output.append('int main() {')
        if main_lines:
            output.extend(main_lines)
        else:
            output.append('    // No executable statements')
        output.append('    return 0;')
        output.append('}')
        return '\n'.join(output)

    def _cpp_indent(self, level: int) -> str:
        return '    ' * level

    def _cpp_current_scope(self) -> set:
        return self._cpp_declared_stack[-1]

    def _cpp_push_scope(self) -> None:
        self._cpp_declared_stack.append(set())
        self._cpp_type_overrides_stack.append({})

    def _cpp_pop_scope(self) -> None:
        if len(self._cpp_declared_stack) > 1:
            self._cpp_declared_stack.pop()
            overrides = self._cpp_type_overrides_stack.pop()
            for name, previous in overrides.items():
                if previous is None:
                    self._cpp_var_types.pop(name, None)
                else:
                    self._cpp_var_types[name] = previous
            self._cpp_vectors = {name for name, t in self._cpp_var_types.items() if t.startswith('vector<')}
            self._cpp_maps = {name for name, t in self._cpp_var_types.items() if 'map<' in t or 'unordered_map<' in t}

    def _cpp_is_declared(self, name: str) -> bool:
        return any(name in scope for scope in self._cpp_declared_stack)

    def _cpp_declare(self, name: str, declared_type: Optional[str] = None) -> None:
        self._cpp_current_scope().add(name)
        if declared_type:
            if name not in self._cpp_type_overrides_stack[-1]:
                self._cpp_type_overrides_stack[-1][name] = self._cpp_var_types.get(name)
            self._cpp_var_types[name] = declared_type
            if declared_type.startswith('vector<'):
                self._cpp_vectors.add(name)
                self._cpp_includes.add('vector')
            if 'map<' in declared_type or 'unordered_map<' in declared_type:
                self._cpp_maps.add(name)

    def _cpp_merge_types(self, types: List[Optional[str]]) -> Optional[str]:
        known = [t for t in types if t and t != 'auto']
        if not known:
            return None
        if any(t.startswith('vector<') for t in known):
            vector_types = [t for t in known if t.startswith('vector<')]
            if len(set(vector_types)) == 1:
                return vector_types[0]
            return 'vector<int>'
        if 'string' in known:
            self._cpp_includes.add('string')
            return 'string'
        if 'double' in known:
            return 'double'
        if 'bool' in known:
            return 'bool'
        if 'int' in known:
            return 'int'
        return known[0]

    def _cpp_normalize_type(self, cpp_type: str) -> str:
        return cpp_type.replace('&', '').replace('const ', '').strip()

    def _cpp_target_mutates_name(self, target: ast.AST, name: str) -> bool:
        if isinstance(target, ast.Name):
            return target.id == name
        if isinstance(target, ast.Subscript):
            return isinstance(target.value, ast.Name) and target.value.id == name
        if isinstance(target, (ast.Tuple, ast.List)):
            return any(self._cpp_target_mutates_name(item, name) for item in target.elts)
        return False

    def _cpp_param_is_mutated(self, func_node: ast.FunctionDef, param_name: str) -> bool:
        mutating_methods = {'append', 'extend', 'insert', 'sort', 'reverse', 'pop', 'clear'}
        for sub in ast.walk(func_node):
            if isinstance(sub, ast.Assign):
                if any(self._cpp_target_mutates_name(target, param_name) for target in sub.targets):
                    return True
            if isinstance(sub, ast.AugAssign):
                if self._cpp_target_mutates_name(sub.target, param_name):
                    return True
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                if isinstance(sub.func.value, ast.Name) and sub.func.value.id == param_name and sub.func.attr in mutating_methods:
                    return True
        return False

    def _cpp_infer_assigned_name_type(self, func_node: ast.FunctionDef, name: str) -> Optional[str]:
        inferred: List[Optional[str]] = []
        for sub in ast.walk(func_node):
            if isinstance(sub, ast.Assign):
                for target in sub.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        inferred.append(self._cpp_infer_type(sub.value))
            elif isinstance(sub, ast.AnnAssign):
                if isinstance(sub.target, ast.Name) and sub.target.id == name and sub.value is not None:
                    inferred.append(self._cpp_infer_type(sub.value))
        return self._cpp_merge_types(inferred)

    def _cpp_infer_param_type(self, func_node: ast.FunctionDef, param_name: str) -> str:
        mutated = self._cpp_param_is_mutated(func_node, param_name)
        default_vector_param = 'vector<int>&' if mutated else 'const vector<int>&'

        for sub in ast.walk(func_node):
            if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Name) and sub.value.id == param_name:
                return default_vector_param
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id == 'len':
                if sub.args and isinstance(sub.args[0], ast.Name) and sub.args[0].id == param_name:
                    return default_vector_param
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                callee = sub.func.id
                if callee in self._cpp_function_param_types:
                    for arg_idx, arg in enumerate(sub.args):
                        if isinstance(arg, ast.Name) and arg.id == param_name and arg_idx < len(self._cpp_function_param_types[callee]):
                            expected = self._cpp_function_param_types[callee][arg_idx]
                            if expected.startswith('vector<'):
                                callee_mutable = False
                                if callee in self._cpp_function_param_mutable and arg_idx < len(self._cpp_function_param_mutable[callee]):
                                    callee_mutable = self._cpp_function_param_mutable[callee][arg_idx]
                                if mutated or callee_mutable:
                                    return f'{expected}&'
                                return f'const {expected}&'
        return 'int'

    def _cpp_detect_vector_append(self, name: str, value: ast.AST) -> Optional[ast.AST]:
        if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Add):
            return None
        if isinstance(value.left, ast.Name) and value.left.id == name and isinstance(value.right, ast.List) and len(value.right.elts) == 1:
            return value.right.elts[0]
        if isinstance(value.right, ast.Name) and value.right.id == name and isinstance(value.left, ast.List) and len(value.left.elts) == 1:
            return value.left.elts[0]
        return None

    def _cpp_emit_dict_assign(self, name: str, dict_node: ast.Dict, indent_level: int) -> List[str]:
        indent = self._cpp_indent(indent_level)
        self._cpp_includes.update({'unordered_map', 'string'})
        map_type = 'unordered_map<string, int>'
        lines: List[str] = []

        entries = []
        for key, value in zip(dict_node.keys, dict_node.values):
            entries.append(f'{{{self._cpp_expr(key)}, {self._cpp_expr(value)}}}')
        init_expr = '{' + ', '.join(entries) + '}'

        if self._cpp_is_declared(name):
            lines.append(f'{indent}{name}.clear();')
            for key, value in zip(dict_node.keys, dict_node.values):
                lines.append(f'{indent}{name}[{self._cpp_expr(key)}] = {self._cpp_expr(value)};')
            return lines

        self._cpp_declare(name, map_type)
        lines.append(f'{indent}{map_type} {name} = {init_expr};')
        return lines

    def _cpp_emit_list_comp_assign(self, name: str, comp_node: ast.ListComp, indent_level: int) -> List[str]:
        indent = self._cpp_indent(indent_level)
        if len(comp_node.generators) != 1:
            return [f'{indent}// Unsupported list comprehension']

        generator = comp_node.generators[0]
        if not isinstance(generator.target, ast.Name):
            return [f'{indent}// Unsupported list comprehension target']

        item_name = generator.target.id
        elem_type = self._cpp_infer_type(comp_node.elt) or 'int'
        if elem_type in ['auto', 'unknown']:
            elem_type = 'int'
        vector_type = f'vector<{elem_type}>'
        self._cpp_includes.add('vector')

        lines: List[str] = []
        if self._cpp_is_declared(name):
            lines.append(f'{indent}{name}.clear();')
        else:
            self._cpp_declare(name, vector_type)
            lines.append(f'{indent}{vector_type} {name};')

        iter_node = generator.iter
        body_indent = self._cpp_indent(indent_level + 1)

        if isinstance(iter_node, ast.Subscript) and isinstance(iter_node.slice, ast.Slice):
            base_expr = self._cpp_expr(iter_node.value)
            start_expr = self._cpp_expr(iter_node.slice.lower) if iter_node.slice.lower else '0'
            stop_expr = self._cpp_expr(iter_node.slice.upper) if iter_node.slice.upper else f'static_cast<int>({base_expr}.size())'
            idx_name = f'__idx{self._cpp_temp_counter}'
            self._cpp_temp_counter += 1
            lines.append(f'{indent}for (int {idx_name} = {start_expr}; {idx_name} < {stop_expr}; ++{idx_name}) {{')
            lines.append(f'{body_indent}{elem_type} {item_name} = {base_expr}[{idx_name}];')
        else:
            iter_expr = self._cpp_expr(iter_node)
            lines.append(f'{indent}for (auto {item_name} : {iter_expr}) {{')

        if generator.ifs:
            cond_expr = ' && '.join(self._cpp_expr(cond) for cond in generator.ifs)
            lines.append(f'{body_indent}if ({cond_expr}) {{')
            lines.append(f'{self._cpp_indent(indent_level + 2)}{name}.push_back({self._cpp_expr(comp_node.elt)});')
            lines.append(f'{body_indent}}}')
        else:
            lines.append(f'{body_indent}{name}.push_back({self._cpp_expr(comp_node.elt)});')
        lines.append(f'{indent}}}')
        return lines

    def _cpp_emit_function(self, node: ast.FunctionDef, indent_level: int) -> List[str]:
        self._cpp_push_scope()
        param_types: Dict[str, str] = {}
        param_mutable: List[bool] = []
        for arg in node.args.args:
            param_mutable.append(self._cpp_param_is_mutated(node, arg.arg))
            param_types[arg.arg] = self._cpp_infer_param_type(node, arg.arg)

        params = []
        for arg in node.args.args:
            param_type = param_types[arg.arg]
            params.append(f'{param_type} {arg.arg}')
            normalized = self._cpp_normalize_type(param_type)
            self._cpp_declare(arg.arg, normalized)
        self._cpp_function_param_types[node.name] = [self._cpp_normalize_type(param_types[arg.arg]) for arg in node.args.args]
        self._cpp_function_param_mutable[node.name] = param_mutable

        return_nodes = [n for n in ast.walk(node) if isinstance(n, ast.Return) and n.value is not None]
        inferred_returns: List[Optional[str]] = []
        for ret in return_nodes:
            inferred = self._cpp_infer_type(ret.value)
            if inferred is None and isinstance(ret.value, ast.Name):
                inferred = self._cpp_infer_assigned_name_type(node, ret.value.id)
            inferred_returns.append(inferred)
        return_type = self._cpp_merge_types(inferred_returns) if inferred_returns else 'void'
        return_type = return_type or 'void'
        self._cpp_function_return_types[node.name] = return_type

        lines = [f'{self._cpp_indent(indent_level)}{return_type} {node.name}({", ".join(params)}) {{']
        body_lines: List[str] = []
        for stmt in node.body:
            body_lines.extend(self._cpp_emit_statement(stmt, indent_level + 1))
        if not body_lines:
            body_lines.append(f'{self._cpp_indent(indent_level + 1)}// No-op')
        lines.extend(body_lines)

        has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
        if return_type != 'void' and not has_return:
            lines.append(f'{self._cpp_indent(indent_level + 1)}return {self._cpp_default_return(return_type)};')

        lines.append(f'{self._cpp_indent(indent_level)}}}')
        self._cpp_pop_scope()
        return lines

    def _cpp_emit_statement(self, stmt: ast.stmt, indent_level: int) -> List[str]:
        indent = self._cpp_indent(indent_level)

        if isinstance(stmt, ast.Assign):
            return self._cpp_emit_assign(stmt, indent_level)

        if isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.value is not None:
                assign_node = ast.Assign(targets=[stmt.target], value=stmt.value)
                return self._cpp_emit_assign(assign_node, indent_level)
            return [f'{indent}// Unsupported annotated assignment']

        if isinstance(stmt, ast.AugAssign):
            target_expr = self._cpp_expr(stmt.target)
            value_expr = self._cpp_expr(stmt.value)
            op = self._cpp_binop(stmt.op)
            return [f'{indent}{target_expr} {op}= {value_expr};']

        if isinstance(stmt, ast.Expr):
            if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == 'print':
                return self._cpp_emit_print(stmt.value, indent_level)
            if isinstance(stmt.value, ast.Call):
                method_lines = self._cpp_emit_collection_call(stmt.value, indent_level)
                if method_lines is not None:
                    return method_lines
            return [f'{indent}{self._cpp_expr(stmt.value)};']

        if isinstance(stmt, ast.For):
            return self._cpp_emit_for(stmt, indent_level)

        if isinstance(stmt, ast.While):
            condition = self._cpp_expr(stmt.test)
            lines = [f'{indent}while ({condition}) {{']
            for inner in stmt.body:
                lines.extend(self._cpp_emit_statement(inner, indent_level + 1))
            lines.append(f'{indent}}}')
            return lines

        if isinstance(stmt, ast.If):
            return self._cpp_emit_if(stmt, indent_level)

        if isinstance(stmt, ast.Match):
            return self._cpp_emit_match(stmt, indent_level)

        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                return [f'{indent}return;']
            return [f'{indent}return {self._cpp_expr(stmt.value)};']

        if isinstance(stmt, ast.Break):
            return [f'{indent}break;']

        if isinstance(stmt, ast.Continue):
            return [f'{indent}continue;']

        if isinstance(stmt, ast.Pass):
            return []

        return [f'{indent}// Unsupported statement: {stmt.__class__.__name__}']

    def _cpp_emit_collection_call(self, call_node: ast.Call, indent_level: int) -> Optional[List[str]]:
        if not isinstance(call_node.func, ast.Attribute):
            return None
        if not isinstance(call_node.func.value, ast.Name):
            return None

        target_name = call_node.func.value.id
        method = call_node.func.attr
        indent = self._cpp_indent(indent_level)

        if method == 'append' and len(call_node.args) == 1:
            return [f'{indent}{target_name}.push_back({self._cpp_expr(call_node.args[0])});']

        if method == 'extend' and len(call_node.args) == 1:
            arg_node = call_node.args[0]
            if isinstance(arg_node, ast.List):
                elem_type = self._cpp_infer_list_element_type(arg_node)
                values = ', '.join(self._cpp_expr(item) for item in arg_node.elts)
                arg_expr = f'vector<{elem_type}>{{{values}}}'
            else:
                arg_expr = self._cpp_expr(arg_node)
            temp_name = f'__ext{self._cpp_temp_counter}'
            self._cpp_temp_counter += 1
            return [
                f'{indent}auto {temp_name} = {arg_expr};',
                f'{indent}{target_name}.insert({target_name}.end(), {temp_name}.begin(), {temp_name}.end());'
            ]

        if method == 'pop' and len(call_node.args) == 0:
            return [f'{indent}if (!{target_name}.empty()) {target_name}.pop_back();']

        return None

    def _cpp_emit_assign(self, stmt: ast.Assign, indent_level: int) -> List[str]:
        indent = self._cpp_indent(indent_level)

        if len(stmt.targets) != 1:
            if all(isinstance(target, ast.Name) for target in stmt.targets):
                value_expr = self._cpp_expr(stmt.value)
                inferred = self._cpp_infer_type(stmt.value)
                lines: List[str] = []
                for target in stmt.targets:
                    name = target.id
                    if self._cpp_is_declared(name):
                        lines.append(f'{indent}{name} = {value_expr};')
                    else:
                        if inferred and inferred != 'auto':
                            self._cpp_declare(name, inferred)
                            lines.append(f'{indent}{inferred} {name} = {value_expr};')
                        else:
                            self._cpp_declare(name, 'auto')
                            lines.append(f'{indent}auto {name} = {value_expr};')
                return lines
            return [f'{indent}// Unsupported multiple-target assignment']

        target = stmt.targets[0]
        value = stmt.value

        if (
            isinstance(target, ast.Tuple)
            and isinstance(value, ast.Tuple)
            and len(target.elts) == 2
            and len(value.elts) == 2
            and self._node_expr_text(value.elts[0]) == self._node_expr_text(target.elts[1])
            and self._node_expr_text(value.elts[1]) == self._node_expr_text(target.elts[0])
        ):
            self._cpp_includes.add('algorithm')
            left_a = self._cpp_expr(target.elts[0])
            left_b = self._cpp_expr(target.elts[1])
            return [f'{indent}swap({left_a}, {left_b});']

        if isinstance(target, ast.Name):
            name = target.id

            if isinstance(value, ast.Dict):
                return self._cpp_emit_dict_assign(name, value, indent_level)

            if isinstance(value, ast.ListComp):
                return self._cpp_emit_list_comp_assign(name, value, indent_level)

            append_expr = self._cpp_detect_vector_append(name, value)
            if append_expr is not None:
                item_expr = self._cpp_expr(append_expr)
                if not self._cpp_is_declared(name):
                    self._cpp_declare(name, 'vector<int>')
                    return [f'{indent}vector<int> {name} = {{{item_expr}}};']
                return [f'{indent}{name}.push_back({item_expr});']

            if isinstance(value, ast.List):
                element_type = self._cpp_infer_list_element_type(value)
                values = ', '.join(self._cpp_expr(item) for item in value.elts)
                if self._cpp_is_declared(name):
                    return [f'{indent}{name} = {{{values}}};']
                declared_type = f'vector<{element_type}>'
                self._cpp_declare(name, declared_type)
                return [f'{indent}{declared_type} {name} = {{{values}}};']

            value_expr = self._cpp_expr(value)
            if self._cpp_is_declared(name):
                return [f'{indent}{name} = {value_expr};']

            inferred = self._cpp_infer_type(value)
            if inferred and inferred != 'auto':
                self._cpp_declare(name, inferred)
                return [f'{indent}{inferred} {name} = {value_expr};']

            self._cpp_declare(name, 'auto')
            return [f'{indent}auto {name} = {value_expr};']

        if isinstance(target, ast.Subscript):
            left = self._cpp_expr(target)
            right = self._cpp_expr(value)
            return [f'{indent}{left} = {right};']

        return [f'{indent}// Unsupported assignment target']

    def _cpp_emit_for(self, stmt: ast.For, indent_level: int) -> List[str]:
        indent = self._cpp_indent(indent_level)
        lines: List[str] = []
        var_name = stmt.target.id if isinstance(stmt.target, ast.Name) else 'item'

        if isinstance(stmt.iter, ast.Call) and isinstance(stmt.iter.func, ast.Name) and stmt.iter.func.id == 'range':
            args = stmt.iter.args
            start_expr = '0'
            stop_expr = '0'
            step_expr = '1'
            step_value = 1

            if len(args) == 1:
                stop_expr = self._cpp_expr(args[0])
            elif len(args) == 2:
                start_expr = self._cpp_expr(args[0])
                stop_expr = self._cpp_expr(args[1])
            elif len(args) >= 3:
                start_expr = self._cpp_expr(args[0])
                stop_expr = self._cpp_expr(args[1])
                step_expr = self._cpp_expr(args[2])
                literal_step = self._int_literal_value(args[2])
                if literal_step is not None:
                    step_value = literal_step

            comparator = '>' if step_value < 0 else '<'
            if step_value == 1:
                update = f'++{var_name}'
            elif step_value == -1:
                update = f'--{var_name}'
            else:
                update = f'{var_name} += {step_expr}'

            lines.append(f'{indent}for (int {var_name} = {start_expr}; {var_name} {comparator} {stop_expr}; {update}) {{')
        else:
            iter_expr = self._cpp_expr(stmt.iter)
            lines.append(f'{indent}for (auto {var_name} : {iter_expr}) {{')

        for inner in stmt.body:
            lines.extend(self._cpp_emit_statement(inner, indent_level + 1))
        lines.append(f'{indent}}}')
        return lines

    def _cpp_emit_if(self, stmt: ast.If, indent_level: int) -> List[str]:
        indent = self._cpp_indent(indent_level)
        branches = []
        else_body: List[ast.stmt] = []
        current = stmt

        while True:
            branches.append((current.test, current.body))
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
                continue
            else_body = current.orelse
            break

        lines: List[str] = []
        for idx, (test_node, body_nodes) in enumerate(branches):
            keyword = 'if' if idx == 0 else 'else if'
            lines.append(f'{indent}{keyword} ({self._cpp_expr(test_node)}) {{')
            for body_stmt in body_nodes:
                lines.extend(self._cpp_emit_statement(body_stmt, indent_level + 1))
            lines.append(f'{indent}}}')

        if else_body:
            lines.append(f'{indent}else {{')
            for body_stmt in else_body:
                lines.extend(self._cpp_emit_statement(body_stmt, indent_level + 1))
            lines.append(f'{indent}}}')

        return lines

    def _cpp_emit_match(self, stmt: ast.Match, indent_level: int) -> List[str]:
        indent = self._cpp_indent(indent_level)
        subject_expr = self._cpp_expr(stmt.subject)
        lines: List[str] = []
        has_branch = False

        for case in stmt.cases:
            values = self._extract_match_pattern_values(case.pattern, self._cpp_expr)
            if values == []:
                header = f'{indent}else {{' if has_branch else f'{indent}{{'
            elif values is not None:
                cond = ' || '.join(f'({subject_expr} == {value})' for value in values)
                header = f'{indent}if ({cond}) {{' if not has_branch else f'{indent}else if ({cond}) {{'
            else:
                lines.append(f'{indent}// Unsupported match pattern')
                continue

            lines.append(header)
            for body_stmt in case.body:
                lines.extend(self._cpp_emit_statement(body_stmt, indent_level + 1))
            lines.append(f'{indent}}}')
            has_branch = True

        if not has_branch:
            return [f'{indent}// Unsupported match statement']
        return lines

    def _cpp_emit_print(self, call_node: ast.Call, indent_level: int) -> List[str]:
        indent = self._cpp_indent(indent_level)
        args = [self._cpp_expr_for_print(arg) for arg in call_node.args]
        if not args:
            return [f'{indent}cout << endl;']
        output = ' << " " << '.join(args)
        return [f'{indent}cout << {output} << endl;']

    def _cpp_expr_for_print(self, node: ast.AST) -> str:
        node_type = self._cpp_infer_type(node)
        if node_type and node_type.startswith('vector<'):
            self._cpp_needs_vector_to_string = True
            return f'vectorToString({self._cpp_expr(node)})'
        if node_type == 'bool':
            expr = self._cpp_expr(node)
            return f'(({expr}) ? "True" : "False")'
        return self._cpp_expr(node)

    def _cpp_expr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, str):
                escaped = value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                return f'"{escaped}"'
            if value is True:
                return 'true'
            if value is False:
                return 'false'
            if value is None:
                return 'nullptr'
            return str(value)

        if isinstance(node, ast.JoinedStr):
            return self._cpp_fstring(node)

        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.BinOp):
            left = self._cpp_expr(node.left)
            right = self._cpp_expr(node.right)
            if isinstance(node.op, ast.Add):
                left_type = self._cpp_infer_type(node.left)
                right_type = self._cpp_infer_type(node.right)
                if (left_type and left_type.startswith('vector<')) or (right_type and right_type.startswith('vector<')):
                    self._cpp_needs_vector_concat_helper = True
                    if not (left_type and left_type.startswith('vector<')):
                        left = f'vector<int>{{{left}}}'
                    if not (right_type and right_type.startswith('vector<')):
                        right = f'vector<int>{{{right}}}'
                    return f'concatVectors({left}, {right})'
            op = self._cpp_binop(node.op)
            return f'({left} {op} {right})'

        if isinstance(node, ast.BoolOp):
            op = ' && ' if isinstance(node.op, ast.And) else ' || '
            return f'({op.join(self._cpp_expr(v) for v in node.values)})'

        if isinstance(node, ast.UnaryOp):
            operand = self._cpp_expr(node.operand)
            if isinstance(node.op, ast.Not):
                return f'(!{operand})'
            if isinstance(node.op, ast.USub):
                return f'(-{operand})'
            return operand

        if isinstance(node, ast.Compare):
            left_expr = self._cpp_expr(node.left)
            parts = []
            for op, comparator in zip(node.ops, node.comparators):
                right_expr = self._cpp_expr(comparator)
                parts.append(f'({left_expr} {self._cpp_compare_op(op)} {right_expr})')
                left_expr = right_expr
            return ' && '.join(parts)

        if isinstance(node, ast.Subscript):
            base_expr = self._cpp_expr(node.value)
            base_type = self._cpp_infer_type(node.value)
            if isinstance(node.slice, ast.Slice):
                step_expr = self._cpp_expr(node.slice.step) if node.slice.step else None
                literal_step = self._int_literal_value(node.slice.step) if node.slice.step else None
                default_start = '0'
                default_end = f'static_cast<int>({base_expr}.size())'
                if literal_step is not None and literal_step < 0:
                    default_start = f'static_cast<int>({base_expr}.size()) - 1'
                    default_end = '-1'
                start_expr = self._cpp_expr(node.slice.lower) if node.slice.lower else default_start
                if base_type == 'string':
                    if node.slice.upper is None:
                        return f'{base_expr}.substr({start_expr})'
                    stop_expr = self._cpp_expr(node.slice.upper)
                    return f'{base_expr}.substr({start_expr}, ({stop_expr}) - ({start_expr}))'
                stop_expr = self._cpp_expr(node.slice.upper) if node.slice.upper else default_end
                if step_expr is not None and (literal_step is None or literal_step != 1):
                    self._cpp_needs_slice_helper = True
                    return f'sliceWithStep({base_expr}, {start_expr}, {stop_expr}, {step_expr})'
                return f'vector<int>({base_expr}.begin() + {start_expr}, {base_expr}.begin() + {stop_expr})'
            return f'{base_expr}[{self._cpp_expr(node.slice)}]'

        if isinstance(node, ast.Attribute):
            return f'{self._cpp_expr(node.value)}.{node.attr}'

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name == 'len' and len(node.args) == 1:
                    return f'static_cast<int>({self._cpp_expr(node.args[0])}.size())'
                if func_name == 'str' and len(node.args) == 1:
                    return f'to_string({self._cpp_expr(node.args[0])})'
                if func_name == 'int' and len(node.args) == 1:
                    return f'static_cast<int>({self._cpp_expr(node.args[0])})'
                if func_name == 'float' and len(node.args) == 1:
                    return f'static_cast<double>({self._cpp_expr(node.args[0])})'
                if func_name in self._cpp_function_return_types:
                    args = ', '.join(self._cpp_expr(arg) for arg in node.args)
                    return f'{func_name}({args})'
            args = ', '.join(self._cpp_expr(arg) for arg in node.args)
            return f'{self._cpp_expr(node.func)}({args})'

        if isinstance(node, ast.List):
            values = ', '.join(self._cpp_expr(item) for item in node.elts)
            return f'{{{values}}}'

        if isinstance(node, ast.Tuple):
            return ', '.join(self._cpp_expr(item) for item in node.elts)

        if isinstance(node, ast.Dict):
            return '/* unsupported dict literal */'

        if isinstance(node, ast.ListComp):
            return '/* unsupported list comprehension */'

        return '/* unsupported */'

    def _cpp_stringify_for_fstring(self, node: ast.AST) -> str:
        expr = self._cpp_expr(node)
        inferred = self._cpp_infer_type(node)

        if inferred and inferred.startswith('vector<'):
            self._cpp_needs_vector_to_string = True
            return f'vectorToString({expr})'
        if inferred == 'string':
            return expr
        if inferred == 'char':
            self._cpp_includes.add('string')
            return f'string(1, {expr})'
        if inferred == 'bool':
            return f'(({expr}) ? "true" : "false")'
        if inferred in ['int', 'double']:
            self._cpp_includes.add('string')
            return f'to_string({expr})'

        self._cpp_includes.update({'string', 'sstream'})
        return f'([&](){{ ostringstream __oss; __oss << {expr}; return __oss.str(); }})()'

    def _cpp_fstring(self, node: ast.JoinedStr) -> str:
        self._cpp_includes.add('string')
        parts: List[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                escaped = value.value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                parts.append(f'"{escaped}"')
            elif isinstance(value, ast.FormattedValue):
                parts.append(self._cpp_stringify_for_fstring(value.value))

        if not parts:
            return '""'
        return ' + '.join(parts)

    def _cpp_binop(self, op_node: ast.AST) -> str:
        mapping = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.FloorDiv: '/',
            ast.Mod: '%',
            ast.Pow: '*'
        }
        for op_type, symbol in mapping.items():
            if isinstance(op_node, op_type):
                return symbol
        return '+'

    def _cpp_compare_op(self, op_node: ast.AST) -> str:
        mapping = {
            ast.Eq: '==',
            ast.NotEq: '!=',
            ast.Lt: '<',
            ast.LtE: '<=',
            ast.Gt: '>',
            ast.GtE: '>='
        }
        for op_type, symbol in mapping.items():
            if isinstance(op_node, op_type):
                return symbol
        return '=='

    def _cpp_infer_list_element_type(self, list_node: ast.List) -> str:
        if not list_node.elts:
            return 'int'
        inferred = [self._cpp_infer_type(item) for item in list_node.elts]
        if any(t == 'string' for t in inferred):
            return 'string'
        if any(t == 'double' for t in inferred):
            return 'double'
        if any(t == 'bool' for t in inferred):
            return 'bool'
        return 'int'

    def _cpp_infer_type(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool):
                return 'bool'
            if isinstance(value, int):
                return 'int'
            if isinstance(value, float):
                return 'double'
            if isinstance(value, str):
                self._cpp_includes.add('string')
                return 'string'
            return None

        if isinstance(node, ast.JoinedStr):
            self._cpp_includes.add('string')
            return 'string'

        if isinstance(node, ast.Name):
            return self._cpp_var_types.get(node.id)

        if isinstance(node, ast.List):
            return f'vector<{self._cpp_infer_list_element_type(node)}>'

        if isinstance(node, ast.Dict):
            self._cpp_includes.update({'unordered_map', 'string'})
            return 'unordered_map<string, int>'

        if isinstance(node, ast.ListComp):
            elem_type = self._cpp_infer_type(node.elt) or 'int'
            elem_type = 'int' if elem_type in ['auto', 'unknown'] else elem_type
            return f'vector<{elem_type}>'

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == 'len':
                return 'int'
            if isinstance(node.func, ast.Name) and node.func.id in self._cpp_function_return_types:
                return self._cpp_function_return_types[node.func.id]
            return None

        if isinstance(node, ast.BoolOp) or isinstance(node, ast.Compare):
            return 'bool'

        if isinstance(node, ast.Subscript):
            parent_type = self._cpp_infer_type(node.value)
            if isinstance(node.slice, ast.Slice):
                if parent_type == 'string':
                    self._cpp_includes.add('string')
                    return 'string'
                if parent_type and parent_type.startswith('vector<'):
                    return parent_type
                return None
            if parent_type and ('unordered_map<' in parent_type or 'map<' in parent_type):
                match = re.search(r'<[^,]+,\s*([^>]+)>', parent_type)
                if match:
                    mapped = match.group(1).strip()
                    mapped = {'Integer': 'int', 'Double': 'double', 'Boolean': 'bool'}.get(mapped, mapped)
                    return mapped
            if parent_type and parent_type.startswith('vector<') and parent_type.endswith('>'):
                return parent_type[7:-1]
            if parent_type == 'string':
                return 'char'
            return None

        if isinstance(node, ast.BinOp):
            left_type = self._cpp_infer_type(node.left)
            right_type = self._cpp_infer_type(node.right)
            if isinstance(node.op, ast.Add):
                if left_type and left_type.startswith('vector<'):
                    return left_type
                if right_type and right_type.startswith('vector<'):
                    return right_type
            if left_type == 'double' or right_type == 'double':
                return 'double'
            if left_type == 'string' or right_type == 'string':
                self._cpp_includes.add('string')
                return 'string'
            return 'int'

        return None

    def _cpp_default_return(self, return_type: str) -> str:
        if return_type == 'int':
            return '0'
        if return_type == 'double':
            return '0.0'
        if return_type == 'bool':
            return 'false'
        if return_type == 'string':
            return '""'
        if return_type.startswith('vector<'):
            return '{}'
        return '{}'

    # -----------------------------
    # Python AST -> JavaScript
    # -----------------------------

    def _convert_python_ast_to_javascript(self, tree: ast.Module) -> str:
        lines: List[str] = []
        for idx, stmt in enumerate(tree.body):
            lines.extend(self._js_emit_statement(stmt, 0))
            if isinstance(stmt, ast.FunctionDef) and idx < len(tree.body) - 1:
                lines.append('')
        return '\n'.join(lines)

    def _js_indent(self, level: int) -> str:
        return '    ' * level

    def _js_current_scope(self) -> set:
        return self._js_declared_stack[-1]

    def _js_push_scope(self) -> None:
        self._js_declared_stack.append(set())

    def _js_pop_scope(self) -> None:
        if len(self._js_declared_stack) > 1:
            self._js_declared_stack.pop()

    def _js_is_declared(self, name: str) -> bool:
        return any(name in scope for scope in self._js_declared_stack)

    def _js_declare(self, name: str) -> None:
        self._js_current_scope().add(name)

    def _js_emit_statement(self, stmt: ast.stmt, indent_level: int) -> List[str]:
        indent = self._js_indent(indent_level)

        if isinstance(stmt, ast.FunctionDef):
            return self._js_emit_function(stmt, indent_level)

        if isinstance(stmt, ast.Assign):
            return self._js_emit_assign(stmt, indent_level)

        if isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, (ast.Name, ast.Subscript, ast.Tuple)) and stmt.value is not None:
                assign_node = ast.Assign(targets=[stmt.target], value=stmt.value)
                return self._js_emit_assign(assign_node, indent_level)
            return [f'{indent}// Unsupported annotated assignment']

        if isinstance(stmt, ast.AugAssign):
            target_expr = self._js_expr(stmt.target)
            value_expr = self._js_expr(stmt.value)
            op = self._js_binop(stmt.op)
            return [f'{indent}{target_expr} {op}= {value_expr};']

        if isinstance(stmt, ast.Expr):
            if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == 'print':
                return self._js_emit_print(stmt.value, indent_level)
            if isinstance(stmt.value, ast.Call):
                method_lines = self._js_emit_collection_call(stmt.value, indent_level)
                if method_lines is not None:
                    return method_lines
            return [f'{indent}{self._js_expr(stmt.value)};']

        if isinstance(stmt, ast.For):
            return self._js_emit_for(stmt, indent_level)

        if isinstance(stmt, ast.While):
            lines = [f'{indent}while ({self._js_expr(stmt.test)}) {{']
            for inner in stmt.body:
                lines.extend(self._js_emit_statement(inner, indent_level + 1))
            lines.append(f'{indent}}}')
            return lines

        if isinstance(stmt, ast.If):
            return self._js_emit_if(stmt, indent_level)

        if isinstance(stmt, ast.Match):
            return self._js_emit_match(stmt, indent_level)

        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                return [f'{indent}return;']
            return [f'{indent}return {self._js_expr(stmt.value)};']

        if isinstance(stmt, ast.Break):
            return [f'{indent}break;']

        if isinstance(stmt, ast.Continue):
            return [f'{indent}continue;']

        if isinstance(stmt, ast.Pass):
            return []

        return [f'{indent}// Unsupported statement: {stmt.__class__.__name__}']

    def _js_emit_collection_call(self, call_node: ast.Call, indent_level: int) -> Optional[List[str]]:
        if not isinstance(call_node.func, ast.Attribute):
            return None
        if not isinstance(call_node.func.value, ast.Name):
            return None

        target_name = call_node.func.value.id
        method = call_node.func.attr
        indent = self._js_indent(indent_level)

        if method == 'append' and len(call_node.args) == 1:
            return [f'{indent}{target_name}.push({self._js_expr(call_node.args[0])});']

        if method == 'extend' and len(call_node.args) == 1:
            return [f'{indent}{target_name}.push(...{self._js_expr(call_node.args[0])});']

        if method == 'pop' and len(call_node.args) == 0:
            return [f'{indent}{target_name}.pop();']

        return None

    def _js_emit_function(self, node: ast.FunctionDef, indent_level: int) -> List[str]:
        indent = self._js_indent(indent_level)
        params = ', '.join(arg.arg for arg in node.args.args)
        self._js_function_return_types[node.name] = self._js_infer_function_return_type(node)

        lines = [f'{indent}function {node.name}({params}) {{']
        self._js_push_scope()
        for arg in node.args.args:
            self._js_declare(arg.arg)
            self._js_var_types[arg.arg] = 'unknown'

        body_lines: List[str] = []
        for inner in node.body:
            body_lines.extend(self._js_emit_statement(inner, indent_level + 1))
        if body_lines:
            lines.extend(body_lines)
        else:
            lines.append(f'{self._js_indent(indent_level + 1)}// No-op')
        lines.append(f'{indent}}}')

        self._js_pop_scope()
        return lines

    def _js_emit_assign(self, stmt: ast.Assign, indent_level: int) -> List[str]:
        indent = self._js_indent(indent_level)
        if len(stmt.targets) != 1:
            if all(isinstance(target, ast.Name) for target in stmt.targets):
                value_expr = self._js_expr(stmt.value)
                inferred_type = self._js_infer_expr_type(stmt.value)
                lines: List[str] = []
                for target in stmt.targets:
                    name = target.id
                    if inferred_type != 'unknown':
                        self._js_var_types[name] = inferred_type
                    if self._js_is_declared(name):
                        lines.append(f'{indent}{name} = {value_expr};')
                    else:
                        self._js_declare(name)
                        lines.append(f'{indent}let {name} = {value_expr};')
                return lines
            return [f'{indent}// Unsupported multiple-target assignment']

        target = stmt.targets[0]
        value = stmt.value

        if (
            isinstance(target, ast.Tuple)
            and isinstance(value, ast.Tuple)
            and len(target.elts) == 2
            and len(value.elts) == 2
            and self._node_expr_text(value.elts[0]) == self._node_expr_text(target.elts[1])
            and self._node_expr_text(value.elts[1]) == self._node_expr_text(target.elts[0])
        ):
            left_a = self._js_expr(target.elts[0])
            left_b = self._js_expr(target.elts[1])
            tmp_name = f'__tmp{self._js_temp_counter}'
            self._js_temp_counter += 1
            return [
                f'{indent}const {tmp_name} = {left_a};',
                f'{indent}{left_a} = {left_b};',
                f'{indent}{left_b} = {tmp_name};'
            ]

        if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple):
            left_items = [self._js_expr(item) for item in target.elts]
            right_items = [self._js_expr(item) for item in value.elts]
            all_names = all(isinstance(item, ast.Name) for item in target.elts)
            need_decl = False
            if all_names:
                for item, rhs in zip(target.elts, value.elts):
                    if not self._js_is_declared(item.id):
                        self._js_declare(item.id)
                        need_decl = True
                    inferred_rhs = self._js_infer_expr_type(rhs)
                    if inferred_rhs != 'unknown':
                        self._js_var_types[item.id] = inferred_rhs
            left = ', '.join(left_items)
            right = ', '.join(right_items)
            if need_decl:
                return [f'{indent}let [{left}] = [{right}];']
            return [f'{indent}[{left}] = [{right}];']

        if isinstance(target, ast.Name):
            value_expr = self._js_expr(value)
            inferred_type = self._js_infer_expr_type(value)
            if inferred_type != 'unknown':
                self._js_var_types[target.id] = inferred_type
            if self._js_is_declared(target.id):
                return [f'{indent}{target.id} = {value_expr};']
            self._js_declare(target.id)
            return [f'{indent}let {target.id} = {value_expr};']

        if isinstance(target, (ast.Subscript, ast.Attribute)):
            return [f'{indent}{self._js_expr(target)} = {self._js_expr(value)};']

        return [f'{indent}// Unsupported assignment target']

    def _js_emit_for(self, stmt: ast.For, indent_level: int) -> List[str]:
        indent = self._js_indent(indent_level)
        var_name = stmt.target.id if isinstance(stmt.target, ast.Name) else '_item'

        if isinstance(stmt.iter, ast.Call) and isinstance(stmt.iter.func, ast.Name) and stmt.iter.func.id == 'range':
            args = stmt.iter.args
            start_expr = '0'
            stop_expr = '0'
            step_expr = '1'
            step_value = 1

            if len(args) == 1:
                stop_expr = self._js_expr(args[0])
            elif len(args) == 2:
                start_expr = self._js_expr(args[0])
                stop_expr = self._js_expr(args[1])
            elif len(args) >= 3:
                start_expr = self._js_expr(args[0])
                stop_expr = self._js_expr(args[1])
                step_expr = self._js_expr(args[2])
                literal_step = self._int_literal_value(args[2])
                if literal_step is not None:
                    step_value = literal_step

            comparator = '>' if step_value < 0 else '<'
            if step_value == 1:
                update = f'{var_name}++'
            elif step_value == -1:
                update = f'{var_name}--'
            else:
                update = f'{var_name} += {step_expr}'

            header = f'{indent}for (let {var_name} = {start_expr}; {var_name} {comparator} {stop_expr}; {update}) {{'
        else:
            iter_expr = self._js_expr(stmt.iter)
            header = f'{indent}for (const {var_name} of {iter_expr}) {{'

        lines = [header]
        self._js_push_scope()
        self._js_declare(var_name)
        for inner in stmt.body:
            lines.extend(self._js_emit_statement(inner, indent_level + 1))
        self._js_pop_scope()
        lines.append(f'{indent}}}')
        return lines

    def _js_emit_if(self, stmt: ast.If, indent_level: int) -> List[str]:
        indent = self._js_indent(indent_level)
        lines: List[str] = []
        current = stmt
        first = True

        while True:
            keyword = 'if' if first else 'else if'
            lines.append(f'{indent}{keyword} ({self._js_expr(current.test)}) {{')
            self._js_push_scope()
            for body_stmt in current.body:
                lines.extend(self._js_emit_statement(body_stmt, indent_level + 1))
            self._js_pop_scope()
            lines.append(f'{indent}}}')

            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
                first = False
                continue
            break

        if current.orelse:
            lines.append(f'{indent}else {{')
            self._js_push_scope()
            for body_stmt in current.orelse:
                lines.extend(self._js_emit_statement(body_stmt, indent_level + 1))
            self._js_pop_scope()
            lines.append(f'{indent}}}')

        return lines

    def _js_emit_match(self, stmt: ast.Match, indent_level: int) -> List[str]:
        indent = self._js_indent(indent_level)
        subject_expr = self._js_expr(stmt.subject)
        lines: List[str] = []
        has_branch = False

        for case in stmt.cases:
            values = self._extract_match_pattern_values(case.pattern, self._js_expr)
            if values == []:
                header = f'{indent}else {{' if has_branch else f'{indent}{{'
            elif values is not None:
                cond = ' || '.join(f'({subject_expr} === {value})' for value in values)
                header = f'{indent}if ({cond}) {{' if not has_branch else f'{indent}else if ({cond}) {{'
            else:
                lines.append(f'{indent}// Unsupported match pattern')
                continue

            lines.append(header)
            for body_stmt in case.body:
                lines.extend(self._js_emit_statement(body_stmt, indent_level + 1))
            lines.append(f'{indent}}}')
            has_branch = True

        if not has_branch:
            return [f'{indent}// Unsupported match statement']
        return lines

    def _js_emit_print(self, call_node: ast.Call, indent_level: int) -> List[str]:
        indent = self._js_indent(indent_level)
        if not call_node.args:
            return [f'{indent}console.log();']
        args = ', '.join(self._js_expr_for_print(arg) for arg in call_node.args)
        return [f'{indent}console.log({args});']

    def _js_expr_for_print(self, node: ast.AST) -> str:
        node_type = self._js_infer_expr_type(node)
        expr = self._js_expr(node)
        if node_type == 'boolean':
            return f'(({expr}) ? "True" : "False")'
        return expr

    def _js_expr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, str):
                escaped = value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                return f'"{escaped}"'
            if value is True:
                return 'true'
            if value is False:
                return 'false'
            if value is None:
                return 'null'
            return str(value)

        if isinstance(node, ast.JoinedStr):
            return self._js_fstring(node)

        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.BinOp):
            left = self._js_expr(node.left)
            right = self._js_expr(node.right)
            if isinstance(node.op, ast.FloorDiv):
                return f'Math.floor(({left}) / ({right}))'
            if isinstance(node.op, ast.Pow):
                return f'Math.pow({left}, {right})'
            if isinstance(node.op, ast.Add):
                left_type = self._js_infer_expr_type(node.left)
                right_type = self._js_infer_expr_type(node.right)
                if left_type == 'array' or right_type == 'array':
                    left_value = left if left_type == 'array' else f'[{left}]'
                    right_value = right if right_type == 'array' else f'[{right}]'
                    return f'({left_value}).concat({right_value})'
            return f'({left} {self._js_binop(node.op)} {right})'

        if isinstance(node, ast.BoolOp):
            op = ' && ' if isinstance(node.op, ast.And) else ' || '
            return f'({op.join(self._js_expr(v) for v in node.values)})'

        if isinstance(node, ast.UnaryOp):
            operand = self._js_expr(node.operand)
            if isinstance(node.op, ast.Not):
                return f'(!{operand})'
            if isinstance(node.op, ast.USub):
                return f'(-{operand})'
            return operand

        if isinstance(node, ast.Compare):
            left_expr = self._js_expr(node.left)
            parts = []
            for op, comparator in zip(node.ops, node.comparators):
                right_expr = self._js_expr(comparator)
                if isinstance(op, ast.In):
                    parts.append(f'({right_expr}.includes({left_expr}))')
                elif isinstance(op, ast.NotIn):
                    parts.append(f'(!{right_expr}.includes({left_expr}))')
                else:
                    parts.append(f'({left_expr} {self._js_compare_op(op)} {right_expr})')
                left_expr = right_expr
            return ' && '.join(parts)

        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Slice):
                return self._js_slice_expr(node.value, node.slice)
            return f'{self._js_expr(node.value)}[{self._js_expr(node.slice)}]'

        if isinstance(node, ast.Attribute):
            return f'{self._js_expr(node.value)}.{node.attr}'

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name == 'len' and len(node.args) == 1:
                    arg_expr = self._js_expr(node.args[0])
                    arg_type = self._js_infer_expr_type(node.args[0])
                    if arg_type == 'object':
                        return f'Object.keys({arg_expr}).length'
                    return f'({arg_expr}).length'
                if func_name == 'str' and len(node.args) == 1:
                    return f'String({self._js_expr(node.args[0])})'
                if func_name == 'int' and len(node.args) == 1:
                    return f'parseInt({self._js_expr(node.args[0])}, 10)'
                if func_name == 'float' and len(node.args) == 1:
                    return f'parseFloat({self._js_expr(node.args[0])})'
                if func_name == 'list' and len(node.args) == 1:
                    return f'Array.from({self._js_expr(node.args[0])})'
            args = ', '.join(self._js_expr(arg) for arg in node.args)
            return f'{self._js_expr(node.func)}({args})'

        if isinstance(node, ast.List):
            return f'[{", ".join(self._js_expr(item) for item in node.elts)}]'

        if isinstance(node, ast.Tuple):
            return f'[{", ".join(self._js_expr(item) for item in node.elts)}]'

        if isinstance(node, ast.Dict):
            pairs = []
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and re.match(r'^[A-Za-z_]\w*$', key.value):
                    pairs.append(f'{key.value}: {self._js_expr(value)}')
                else:
                    pairs.append(f'[{self._js_expr(key)}]: {self._js_expr(value)}')
            return f'{{{", ".join(pairs)}}}'

        if isinstance(node, ast.ListComp):
            return self._js_list_comp_expr(node)

        if isinstance(node, ast.IfExp):
            return f'({self._js_expr(node.test)} ? {self._js_expr(node.body)} : {self._js_expr(node.orelse)})'

        return 'null'

    def _js_fstring(self, node: ast.JoinedStr) -> str:
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                text = value.value.replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${')
                parts.append(text)
            elif isinstance(value, ast.FormattedValue):
                parts.append('${' + self._js_expr(value.value) + '}')
        return '`' + ''.join(parts) + '`'

    def _js_slice_expr(self, value_node: ast.AST, slice_node: ast.Slice) -> str:
        base = self._js_expr(value_node)
        start = self._js_expr(slice_node.lower) if slice_node.lower is not None else None
        stop = self._js_expr(slice_node.upper) if slice_node.upper is not None else None
        step = self._js_expr(slice_node.step) if slice_node.step is not None else None

        if start is None and stop is None:
            sliced = f'({base}).slice()'
        elif stop is None:
            sliced = f'({base}).slice({start})'
        elif start is None:
            sliced = f'({base}).slice(0, {stop})'
        else:
            sliced = f'({base}).slice({start}, {stop})'

        if step is None or step == '1':
            return sliced

        if isinstance(slice_node.step, ast.Constant) and isinstance(slice_node.step.value, int):
            step_value = slice_node.step.value
            if step_value > 1:
                return f'{sliced}.filter((_, idx) => idx % {step_value} === 0)'
            if step_value < 0:
                return f'{sliced}.reverse().filter((_, idx) => idx % {abs(step_value)} === 0)'

        return f'{sliced}.filter((_, idx) => idx % ({step}) === 0)'

    def _js_list_comp_expr(self, node: ast.ListComp) -> str:
        if len(node.generators) != 1:
            return '[]'

        generator = node.generators[0]
        target = generator.target.id if isinstance(generator.target, ast.Name) else '_item'
        base_iter = f'Array.from({self._js_expr(generator.iter)})'
        elt_expr = self._js_expr(node.elt)
        identity = isinstance(node.elt, ast.Name) and node.elt.id == target

        if generator.ifs:
            cond = ' && '.join(self._js_expr(cond_node) for cond_node in generator.ifs)
            filtered = f'{base_iter}.filter(({target}) => {cond})'
            if identity:
                return filtered
            return f'{filtered}.map(({target}) => {elt_expr})'

        if identity:
            return base_iter
        return f'{base_iter}.map(({target}) => {elt_expr})'

    def _js_infer_function_return_type(self, node: ast.FunctionDef) -> str:
        return_types = []
        for found in ast.walk(node):
            if isinstance(found, ast.Return) and found.value is not None:
                return_types.append(self._js_infer_expr_type(found.value))

        known = [t for t in return_types if t != 'unknown']
        if not known:
            return 'unknown'
        if 'array' in known:
            return 'array'
        if all(t == known[0] for t in known):
            return known[0]
        return 'unknown'

    def _js_infer_expr_type(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool):
                return 'boolean'
            if isinstance(value, (int, float)):
                return 'number'
            if isinstance(value, str):
                return 'string'
            if value is None:
                return 'null'
            return 'unknown'

        if isinstance(node, ast.JoinedStr):
            return 'string'

        if isinstance(node, ast.Name):
            return self._js_var_types.get(node.id, 'unknown')

        if isinstance(node, (ast.List, ast.Tuple, ast.ListComp)):
            return 'array'

        if isinstance(node, ast.Dict):
            return 'object'

        if isinstance(node, ast.BoolOp) or isinstance(node, ast.Compare):
            return 'boolean'

        if isinstance(node, ast.IfExp):
            left = self._js_infer_expr_type(node.body)
            right = self._js_infer_expr_type(node.orelse)
            if left == right:
                return left
            if left == 'unknown':
                return right
            if right == 'unknown':
                return left
            return 'unknown'

        if isinstance(node, ast.Subscript):
            base_type = self._js_infer_expr_type(node.value)
            if base_type == 'string':
                return 'string'
            return 'unknown'

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in ['len', 'int', 'float']:
                    return 'number'
                if func_name == 'str':
                    return 'string'
                if func_name == 'list':
                    return 'array'
                if func_name in self._js_function_return_types:
                    return self._js_function_return_types[func_name]
            return 'unknown'

        if isinstance(node, ast.BinOp):
            left_type = self._js_infer_expr_type(node.left)
            right_type = self._js_infer_expr_type(node.right)
            if isinstance(node.op, ast.Add):
                if left_type == 'array' or right_type == 'array':
                    return 'array'
                if left_type == 'string' or right_type == 'string':
                    return 'string'
                if left_type == 'number' and right_type == 'number':
                    return 'number'
                return 'unknown'
            if isinstance(node.op, (ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv, ast.Pow)):
                return 'number'

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return 'boolean'
            if isinstance(node.op, ast.USub):
                return 'number'

        return 'unknown'

    def _js_binop(self, op_node: ast.AST) -> str:
        mapping = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.Mod: '%'
        }
        for op_type, symbol in mapping.items():
            if isinstance(op_node, op_type):
                return symbol
        return '+'

    def _js_compare_op(self, op_node: ast.AST) -> str:
        mapping = {
            ast.Eq: '===',
            ast.NotEq: '!==',
            ast.Lt: '<',
            ast.LtE: '<=',
            ast.Gt: '>',
            ast.GtE: '>='
        }
        for op_type, symbol in mapping.items():
            if isinstance(op_node, op_type):
                return symbol
        return '==='

    # -----------------------------
    # Python AST -> Ruby
    # -----------------------------

    def _convert_python_ast_to_ruby(self, tree: ast.Module) -> str:
        lines: List[str] = []
        for idx, stmt in enumerate(tree.body):
            lines.extend(self._ruby_emit_statement(stmt, 0))
            if isinstance(stmt, ast.FunctionDef) and idx < len(tree.body) - 1:
                lines.append('')
        return '\n'.join(lines)

    def _ruby_indent(self, level: int) -> str:
        return '  ' * level

    def _ruby_emit_statement(self, stmt: ast.stmt, indent_level: int) -> List[str]:
        indent = self._ruby_indent(indent_level)

        if isinstance(stmt, ast.FunctionDef):
            params = ', '.join(arg.arg for arg in stmt.args.args)
            lines = [f'{indent}def {stmt.name}({params})']
            body_lines: List[str] = []
            for inner in stmt.body:
                body_lines.extend(self._ruby_emit_statement(inner, indent_level + 1))
            if body_lines:
                lines.extend(body_lines)
            else:
                lines.append(f'{self._ruby_indent(indent_level + 1)}# No-op')
            lines.append(f'{indent}end')
            return lines

        if isinstance(stmt, ast.Assign):
            return self._ruby_emit_assign(stmt, indent_level)

        if isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, (ast.Name, ast.Subscript, ast.Tuple)) and stmt.value is not None:
                assign_node = ast.Assign(targets=[stmt.target], value=stmt.value)
                return self._ruby_emit_assign(assign_node, indent_level)
            return [f'{indent}# Unsupported annotated assignment']

        if isinstance(stmt, ast.AugAssign):
            target_expr = self._ruby_expr(stmt.target)
            value_expr = self._ruby_expr(stmt.value)
            op = self._ruby_binop(stmt.op)
            return [f'{indent}{target_expr} {op}= {value_expr}']

        if isinstance(stmt, ast.Expr):
            if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == 'print':
                return self._ruby_emit_print(stmt.value, indent_level)
            if isinstance(stmt.value, ast.Call):
                method_line = self._ruby_emit_collection_call(stmt.value, indent_level)
                if method_line is not None:
                    return method_line
            return [f'{indent}{self._ruby_expr(stmt.value)}']

        if isinstance(stmt, ast.For):
            return self._ruby_emit_for(stmt, indent_level)

        if isinstance(stmt, ast.While):
            lines = [f'{indent}while {self._ruby_expr(stmt.test)}']
            for inner in stmt.body:
                lines.extend(self._ruby_emit_statement(inner, indent_level + 1))
            lines.append(f'{indent}end')
            return lines

        if isinstance(stmt, ast.If):
            return self._ruby_emit_if(stmt, indent_level)

        if isinstance(stmt, ast.Match):
            return self._ruby_emit_match(stmt, indent_level)

        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                return [f'{indent}return']
            return [f'{indent}return {self._ruby_expr(stmt.value)}']

        if isinstance(stmt, ast.Break):
            return [f'{indent}break']

        if isinstance(stmt, ast.Continue):
            return [f'{indent}next']

        if isinstance(stmt, ast.Pass):
            return []

        return [f'{indent}# Unsupported statement: {stmt.__class__.__name__}']

    def _ruby_emit_collection_call(self, call_node: ast.Call, indent_level: int) -> Optional[List[str]]:
        if not isinstance(call_node.func, ast.Attribute):
            return None
        if not isinstance(call_node.func.value, ast.Name):
            return None

        target_name = call_node.func.value.id
        method = call_node.func.attr
        indent = self._ruby_indent(indent_level)

        if method == 'append' and len(call_node.args) == 1:
            return [f'{indent}{target_name} << {self._ruby_expr(call_node.args[0])}']

        if method == 'extend' and len(call_node.args) == 1:
            return [f'{indent}{target_name}.concat({self._ruby_expr(call_node.args[0])})']

        if method == 'pop' and len(call_node.args) == 0:
            return [f'{indent}{target_name}.pop']

        return None

    def _ruby_emit_assign(self, stmt: ast.Assign, indent_level: int) -> List[str]:
        indent = self._ruby_indent(indent_level)

        if len(stmt.targets) != 1:
            if all(isinstance(target, ast.Name) for target in stmt.targets):
                value_expr = self._ruby_expr(stmt.value)
                return [f'{indent}{target.id} = {value_expr}' for target in stmt.targets]
            return [f'{indent}# Unsupported multiple-target assignment']

        target = stmt.targets[0]
        value = stmt.value

        if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple):
            left = ', '.join(self._ruby_expr(item) for item in target.elts)
            right = ', '.join(self._ruby_expr(item) for item in value.elts)
            return [f'{indent}{left} = {right}']

        if isinstance(target, (ast.Name, ast.Subscript, ast.Attribute)):
            return [f'{indent}{self._ruby_expr(target)} = {self._ruby_expr(value)}']

        return [f'{indent}# Unsupported assignment target']

    def _ruby_emit_for(self, stmt: ast.For, indent_level: int) -> List[str]:
        indent = self._ruby_indent(indent_level)
        loop_var = stmt.target.id if isinstance(stmt.target, ast.Name) else '_item'

        if isinstance(stmt.iter, ast.Call) and isinstance(stmt.iter.func, ast.Name) and stmt.iter.func.id == 'range':
            args = stmt.iter.args
            if len(args) == 1:
                start_expr = '0'
                stop_expr = self._ruby_expr(args[0])
                step_expr = None
            elif len(args) >= 2:
                start_expr = self._ruby_expr(args[0])
                stop_expr = self._ruby_expr(args[1])
                step_expr = self._ruby_expr(args[2]) if len(args) >= 3 else None
            else:
                start_expr = '0'
                stop_expr = '0'
                step_expr = None

            range_expr = f'({start_expr}...{stop_expr})'
            if step_expr is None or step_expr == '1':
                header = f'{indent}{range_expr}.each do |{loop_var}|'
            else:
                header = f'{indent}{range_expr}.step({step_expr}).each do |{loop_var}|'
        else:
            iter_expr = self._ruby_expr(stmt.iter)
            header = f'{indent}{iter_expr}.each do |{loop_var}|'

        lines = [header]
        for inner in stmt.body:
            lines.extend(self._ruby_emit_statement(inner, indent_level + 1))
        lines.append(f'{indent}end')
        return lines

    def _ruby_emit_if(self, stmt: ast.If, indent_level: int) -> List[str]:
        indent = self._ruby_indent(indent_level)
        lines: List[str] = []
        current = stmt
        first = True

        while True:
            keyword = 'if' if first else 'elsif'
            lines.append(f'{indent}{keyword} {self._ruby_expr(current.test)}')
            for body_stmt in current.body:
                lines.extend(self._ruby_emit_statement(body_stmt, indent_level + 1))

            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                current = current.orelse[0]
                first = False
                continue
            break

        if current.orelse:
            lines.append(f'{indent}else')
            for body_stmt in current.orelse:
                lines.extend(self._ruby_emit_statement(body_stmt, indent_level + 1))

        lines.append(f'{indent}end')
        return lines

    def _ruby_emit_match(self, stmt: ast.Match, indent_level: int) -> List[str]:
        indent = self._ruby_indent(indent_level)
        subject_expr = self._ruby_expr(stmt.subject)
        lines: List[str] = [f'{indent}case {subject_expr}']
        has_branch = False

        for case in stmt.cases:
            values = self._extract_match_pattern_values(case.pattern, self._ruby_expr)
            if values == []:
                lines.append(f'{indent}else')
            elif values is not None:
                lines.append(f'{indent}when {", ".join(values)}')
            else:
                lines.append(f'{indent}# Unsupported match pattern')
                continue

            for body_stmt in case.body:
                lines.extend(self._ruby_emit_statement(body_stmt, indent_level + 1))
            has_branch = True

        lines.append(f'{indent}end')
        if not has_branch:
            return [f'{indent}# Unsupported match statement']
        return lines

    def _ruby_emit_print(self, call_node: ast.Call, indent_level: int) -> List[str]:
        indent = self._ruby_indent(indent_level)
        if not call_node.args:
            return [f'{indent}puts']

        arg_exprs = ', '.join(self._ruby_expr(arg) for arg in call_node.args)
        return [f'{indent}puts [{arg_exprs}].map {{ |v| v.is_a?(Array) ? v.inspect : (v == true ? "True" : (v == false ? "False" : v.to_s)) }}.join(" ")']

    def _ruby_expr(self, node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, str):
                escaped = value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                return f'"{escaped}"'
            if value is True:
                return 'true'
            if value is False:
                return 'false'
            if value is None:
                return 'nil'
            return str(value)

        if isinstance(node, ast.JoinedStr):
            return self._ruby_fstring(node)

        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.BinOp):
            left = self._ruby_expr(node.left)
            right = self._ruby_expr(node.right)
            if isinstance(node.op, ast.FloorDiv):
                return f'(({left}) / ({right}))'
            op = self._ruby_binop(node.op)
            return f'({left} {op} {right})'

        if isinstance(node, ast.BoolOp):
            op = ' && ' if isinstance(node.op, ast.And) else ' || '
            return f'({op.join(self._ruby_expr(v) for v in node.values)})'

        if isinstance(node, ast.UnaryOp):
            operand = self._ruby_expr(node.operand)
            if isinstance(node.op, ast.Not):
                return f'(!{operand})'
            if isinstance(node.op, ast.USub):
                return f'(-{operand})'
            return operand

        if isinstance(node, ast.Compare):
            left_expr = self._ruby_expr(node.left)
            parts = []
            for op, comparator in zip(node.ops, node.comparators):
                right_expr = self._ruby_expr(comparator)
                parts.append(f'({left_expr} {self._ruby_compare_op(op)} {right_expr})')
                left_expr = right_expr
            return ' && '.join(parts)

        if isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Slice):
                return self._ruby_slice_expr(node.value, node.slice)
            return f'{self._ruby_expr(node.value)}[{self._ruby_expr(node.slice)}]'

        if isinstance(node, ast.Attribute):
            return f'{self._ruby_expr(node.value)}.{node.attr}'

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name == 'len' and len(node.args) == 1:
                    return f'({self._ruby_expr(node.args[0])}).length'
                if func_name == 'str' and len(node.args) == 1:
                    return f'({self._ruby_expr(node.args[0])}).to_s'
                if func_name == 'int' and len(node.args) == 1:
                    return f'({self._ruby_expr(node.args[0])}).to_i'
                if func_name == 'float' and len(node.args) == 1:
                    return f'({self._ruby_expr(node.args[0])}).to_f'
                if func_name == 'list' and len(node.args) == 1:
                    return f'({self._ruby_expr(node.args[0])}).to_a'
            args = ', '.join(self._ruby_expr(arg) for arg in node.args)
            return f'{self._ruby_expr(node.func)}({args})'

        if isinstance(node, ast.List):
            return f'[{", ".join(self._ruby_expr(item) for item in node.elts)}]'

        if isinstance(node, ast.Tuple):
            return f'[{", ".join(self._ruby_expr(item) for item in node.elts)}]'

        if isinstance(node, ast.Dict):
            pairs = []
            for key, value in zip(node.keys, node.values):
                pairs.append(f'{self._ruby_expr(key)} => {self._ruby_expr(value)}')
            return f'{{{", ".join(pairs)}}}'

        if isinstance(node, ast.ListComp):
            return self._ruby_list_comp_expr(node)

        return 'nil'

    def _ruby_fstring(self, node: ast.JoinedStr) -> str:
        parts: List[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                text = value.value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('#{', '\\#{')
                parts.append(text)
            elif isinstance(value, ast.FormattedValue):
                parts.append(f'#{{{self._ruby_expr(value.value)}}}')
        return '"' + ''.join(parts) + '"'

    def _ruby_slice_expr(self, value_node: ast.AST, slice_node: ast.Slice) -> str:
        base = self._ruby_expr(value_node)
        start = self._ruby_expr(slice_node.lower) if slice_node.lower is not None else None
        stop = self._ruby_expr(slice_node.upper) if slice_node.upper is not None else None
        step = self._ruby_expr(slice_node.step) if slice_node.step is not None else None

        if start is not None and stop is not None:
            range_expr = f'{start}...{stop}'
        elif start is not None:
            range_expr = f'{start}..'
        elif stop is not None:
            range_expr = f'0...{stop}'
        else:
            range_expr = '0..'

        sliced = f'({base}[{range_expr}] || [])'

        if step is None or step == '1':
            return sliced

        if isinstance(slice_node.step, ast.Constant) and isinstance(slice_node.step.value, int):
            step_val = slice_node.step.value
            if step_val > 1:
                return f'{sliced}.each_slice({step_val}).map(&:first)'
            if step_val < 0:
                return f'{sliced}.reverse.each_slice({abs(step_val)}).map(&:first)'

        return f'{sliced}.each_with_index.select {{ |_, idx| idx % ({step}) == 0 }}.map(&:first)'

    def _ruby_list_comp_expr(self, node: ast.ListComp) -> str:
        if len(node.generators) != 1:
            return '[]'

        generator = node.generators[0]
        target_name = generator.target.id if isinstance(generator.target, ast.Name) else '_item'
        iter_expr = self._ruby_expr(generator.iter)

        cond_expr = None
        if generator.ifs:
            cond_expr = ' && '.join(self._ruby_expr(cond) for cond in generator.ifs)

        elt_expr = self._ruby_expr(node.elt)
        identity = isinstance(node.elt, ast.Name) and node.elt.id == target_name

        if cond_expr:
            filtered = f'({iter_expr}).select {{ |{target_name}| {cond_expr} }}'
            if identity:
                return filtered
            return f'{filtered}.map {{ |{target_name}| {elt_expr} }}'

        if identity:
            return f'({iter_expr}).to_a'
        return f'({iter_expr}).map {{ |{target_name}| {elt_expr} }}'

    def _ruby_binop(self, op_node: ast.AST) -> str:
        mapping = {
            ast.Add: '+',
            ast.Sub: '-',
            ast.Mult: '*',
            ast.Div: '/',
            ast.Mod: '%',
            ast.Pow: '**'
        }
        for op_type, symbol in mapping.items():
            if isinstance(op_node, op_type):
                return symbol
        return '+'

    def _ruby_compare_op(self, op_node: ast.AST) -> str:
        mapping = {
            ast.Eq: '==',
            ast.NotEq: '!=',
            ast.Lt: '<',
            ast.LtE: '<=',
            ast.Gt: '>',
            ast.GtE: '>='
        }
        for op_type, symbol in mapping.items():
            if isinstance(op_node, op_type):
                return symbol
        return '=='
    
    def _convert_code(self, code: str, source: str, target: str) -> str:
        """Main conversion logic"""
        lines = code.split('\n')
        result_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())
            
            if not stripped:
                i += 1
                continue
            
            # Comments
            if stripped.startswith('#'):
                comment = stripped[1:].strip()
                if target in ['python', 'ruby']:
                    result_lines.append(' ' * indent + f'# {comment}')
                else:
                    result_lines.append(' ' * indent + f'// {comment}')
                i += 1
                continue
            
            # Function definition
            if self._is_function_def(stripped, source):
                func_lines, i = self._extract_block(lines, i, source)
                converted_func = self._convert_function(func_lines, source, target)
                result_lines.extend(converted_func)
                continue
            
            # For loop
            if self._is_for_loop(stripped, source):
                loop_lines, i = self._extract_block(lines, i, source)
                converted_loop = self._convert_for_loop(loop_lines, source, target)
                result_lines.extend(converted_loop)
                continue
            
            # While loop
            if self._is_while_loop(stripped, source):
                loop_lines, i = self._extract_block(lines, i, source)
                converted_loop = self._convert_while_loop(loop_lines, source, target)
                result_lines.extend(converted_loop)
                continue
            
            # If statement
            if self._is_if_statement(stripped, source):
                if_lines, i = self._extract_block(lines, i, source)
                converted_if = self._convert_if(if_lines, source, target)
                result_lines.extend(converted_if)
                continue
            
            # Print statement
            if self._is_print(stripped, source):
                converted = self._convert_print(stripped, source, target, indent)
                result_lines.append(converted)
                i += 1
                continue
            
            # Variable assignment
            if self._is_assignment(stripped, source):
                converted = self._convert_assignment(stripped, source, target, indent)
                result_lines.append(converted)
                i += 1
                continue
            
            # Array/List operations
            if self._is_array_op(stripped, source):
                converted = self._convert_array_op(stripped, source, target, indent)
                result_lines.append(converted)
                i += 1
                continue
            
            # Return statement
            if stripped.startswith('return'):
                converted = self._convert_return(stripped, source, target, indent)
                result_lines.append(converted)
                i += 1
                continue
            
            # Generic line conversion
            converted = self._convert_generic_line(stripped, source, target, indent)
            result_lines.append(converted)
            i += 1
        
        return '\n'.join(result_lines)
    
    def _is_function_def(self, line: str, lang: str) -> bool:
        patterns = {
            'python': r'^def\s+\w+',
            'javascript': r'^function\s+\w+|^(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\(',
            'java': r'^(?:public|private|protected|static)?\s*(?:\w+\s+)+\w+\s*\(',
            'cpp': r'^(?:\w+\s+)+\w+\s*\(',
            'ruby': r'^def\s+\w+'
        }
        return bool(re.match(patterns.get(lang, ''), line))
    
    def _is_for_loop(self, line: str, lang: str) -> bool:
        patterns = {
            'python': r'^for\s+',
            'javascript': r'^for\s*\(',
            'java': r'^for\s*\(',
            'cpp': r'^for\s*\(',
            'ruby': r'^\w+\.each\s+do|^\d+\.times\s+do|^for\s+'
        }
        return bool(re.match(patterns.get(lang, ''), line))
    
    def _is_while_loop(self, line: str, lang: str) -> bool:
        patterns = {
            'python': r'^while\s+',
            'javascript': r'^while\s*\(',
            'java': r'^while\s*\(',
            'cpp': r'^while\s*\(',
            'ruby': r'^while\s+'
        }
        return bool(re.match(patterns.get(lang, ''), line))
    
    def _is_if_statement(self, line: str, lang: str) -> bool:
        patterns = {
            'python': r'^if\s+|^elif\s+|^else:',
            'javascript': r'^if\s*\(|^else\s*\{|^else\s+if',
            'java': r'^if\s*\(|^else\s*\{|^else\s+if',
            'cpp': r'^if\s*\(|^else\s*\{|^else\s+if',
            'ruby': r'^if\s+|^elsif\s+|^else$'
        }
        return bool(re.match(patterns.get(lang, ''), line))
    
    def _is_print(self, line: str, lang: str) -> bool:
        patterns = {
            'python': r'^print\s*\(',
            'javascript': r'^console\.log\s*\(',
            'java': r'^System\.out\.print',
            'cpp': r'^(?:std::)?cout\s*<<',
            'ruby': r'^puts\s+|^print\s+'
        }
        return bool(re.match(patterns.get(lang, ''), line))
    
    def _is_assignment(self, line: str, lang: str) -> bool:
        patterns = {
            'python': r'^[a-zA-Z_]\w*\s*=(?!=)',
            'javascript': r'^(?:let|const|var)\s+[a-zA-Z_]',
            'java': r'^(?:int|String|double|float|boolean|char|long|var)\s+[a-zA-Z_]',
            'cpp': r'^(?:int|double|float|char|string|auto|void)\s+[a-zA-Z_]',
            'ruby': r'^[a-zA-Z_]\w*\s*=(?!=)'
        }
        return bool(re.match(patterns.get(lang, ''), line))
    
    def _is_array_op(self, line: str, lang: str) -> bool:
        patterns = {
            'python': r'.*\[\s*:\s*\]|.*\.append\(|.*\.pop\(|.*\.sort\(|.*\.reverse\(',
            'javascript': r'.*\.push\(|.*\.pop\(|.*\.sort\(|.*\.reverse\(',
            'java': r'.*\.add\(|.*\.remove\(|.*\.sort\(',
            'cpp': r'.*\.push_back\(|.*\.pop_back\(|.*\.sort\(',
            'ruby': r'.*\.push\(|.*\.pop\(|.*\.sort\(|.*\.reverse\('
        }
        return bool(re.search(patterns.get(lang, ''), line))
    
    def _extract_block(self, lines: List[str], start: int, lang: str) -> Tuple[List[str], int]:
        """Extract a code block (function, loop, if)"""
        block_lines = [lines[start]]
        i = start + 1
        
        if lang in ['python', 'ruby']:
            # Python uses indentation
            base_indent = len(lines[start]) - len(lines[start].lstrip())
            while i < len(lines):
                line = lines[i]
                if line.strip() == '':
                    block_lines.append(line)
                    i += 1
                    continue
                current_indent = len(line) - len(line.lstrip())
                if current_indent > base_indent:
                    block_lines.append(line)
                    i += 1
                else:
                    break
        else:
            # C-style uses braces
            brace_count = lines[start].count('{') - lines[start].count('}')
            while i < len(lines) and brace_count > 0:
                block_lines.append(lines[i])
                brace_count += lines[i].count('{') - lines[i].count('}')
                i += 1
        
        return block_lines, i
    
    def _convert_function(self, lines: List[str], source: str, target: str) -> List[str]:
        """Convert function definition"""
        first_line = lines[0].strip()
        result = []
        indent = len(lines[0]) - len(lines[0].lstrip())
        
        # Extract function name and params
        if source == 'python':
            match = re.match(r'def\s+(\w+)\s*\(([^)]*)\)', first_line)
            if match:
                name, params = match.groups()
                params = [p.strip().split(':')[0].strip() for p in params.split(',') if p.strip()]
        elif source == 'ruby':
            match = re.match(r'def\s+(\w+)\s*(?:\(([^)]*)\))?', first_line)
            if match:
                name = match.group(1)
                params_str = match.group(2) or ''
                params = [p.strip() for p in params_str.split(',') if p.strip()]
        elif source in ['javascript', 'java', 'cpp']:
            match = re.search(r'(\w+)\s*\(([^)]*)\)', first_line)
            if match:
                name = match.group(1)
                params_str = match.group(2)
                params = [p.strip().split()[-1] for p in params_str.split(',') if p.strip()]
        else:
            name, params = 'func', []
        
        # Generate function header
        params_str = ', '.join(params) if params else ''
        
        if target == 'python':
            result.append(' ' * indent + f'def {name}({params_str}):')
        elif target == 'javascript':
            result.append(' ' * indent + f'function {name}({params_str}) {{')
        elif target == 'java':
            param_list = ', '.join([f'int {p}' for p in params]) if params else ''
            result.append(' ' * indent + f'public static void {name}({param_list}) {{')
        elif target == 'cpp':
            param_list = ', '.join([f'int {p}' for p in params]) if params else ''
            result.append(' ' * indent + f'void {name}({param_list}) {{')
        elif target == 'ruby':
            result.append(' ' * indent + f'def {name}({params_str})')
        
        # Convert body
        body_lines = lines[1:] if len(lines) > 1 else []
        for line in body_lines:
            if line.strip():
                converted = self._convert_code(line.strip(), source, target)
                body_indent = len(line) - len(line.lstrip())
                if target in ['python']:
                    result.append(' ' * (indent + 4) + converted.strip())
                elif target in ['javascript', 'java', 'cpp']:
                    result.append(' ' * (indent + 4) + converted.strip())
                elif target == 'ruby':
                    result.append(' ' * (indent + 2) + converted.strip())
            else:
                result.append('')
        
        # Close block
        if target in ['javascript', 'java', 'cpp']:
            result.append(' ' * indent + '}')
        elif target == 'ruby':
            result.append(' ' * indent + 'end')
        
        return result
    
    def _convert_for_loop(self, lines: List[str], source: str, target: str) -> List[str]:
        """Convert for loop"""
        first_line = lines[0].strip()
        result = []
        indent = len(lines[0]) - len(lines[0].lstrip())
        
        # Parse for loop
        var, start_val, end_val, step = 'i', 0, 10, 1
        
        if source == 'python':
            # for i in range(n): or for i in range(start, end):
            match = re.match(r'for\s+(\w+)\s+in\s+range\s*\(([^)]+)\)', first_line)
            if match:
                var = match.group(1)
                range_args = match.group(2).split(',')
                if len(range_args) == 1:
                    end_val = range_args[0].strip()
                elif len(range_args) == 2:
                    start_val = range_args[0].strip()
                    end_val = range_args[1].strip()
                elif len(range_args) == 3:
                    start_val = range_args[0].strip()
                    end_val = range_args[1].strip()
                    step = range_args[2].strip()
            
            # for item in list:
            match = re.match(r'for\s+(\w+)\s+in\s+(\w+)', first_line)
            if match and 'range' not in first_line:
                var = match.group(1)
                list_name = match.group(2)
                # Convert to iteration
                if target == 'python':
                    result.append(' ' * indent + f'for {var} in {list_name}:')
                elif target == 'javascript':
                    result.append(' ' * indent + f'for (let {var} of {list_name}) {{')
                elif target in ['java', 'cpp']:
                    result.append(' ' * indent + f'for (auto {var} : {list_name}) {{')
                elif target == 'ruby':
                    result.append(' ' * indent + f'{list_name}.each do |{var}|')
                
                # Process body
                for line in lines[1:]:
                    if line.strip():
                        converted = self._convert_code(line.strip(), source, target)
                        result.append(' ' * (indent + 4) + converted.strip())
                
                if target in ['javascript', 'java', 'cpp']:
                    result.append(' ' * indent + '}')
                elif target == 'ruby':
                    result.append(' ' * indent + 'end')
                return result
        elif source in ['javascript', 'java', 'cpp']:
            c_match = re.match(r'for\s*\((.+)\)\s*\{?', first_line)
            if c_match:
                c_header = c_match.group(1).strip()
                if target in ['javascript', 'java', 'cpp']:
                    result.append(' ' * indent + f'for ({c_header}) {{')
                elif target == 'ruby':
                    result.append(' ' * indent + f'for ({c_header})')
                elif target == 'python':
                    result.append(' ' * indent + '# Unsupported C-style for conversion')
                    for line in lines[1:]:
                        if line.strip():
                            converted = self._convert_generic_line(line.strip(), source, target, indent + 4)
                            result.append(converted)
                    return result

                for line in lines[1:]:
                    stripped_body = line.strip()
                    if not stripped_body or stripped_body in ['{', '}']:
                        continue
                    converted = self._convert_generic_line(stripped_body, source, target, indent + 4)
                    result.append(converted)

                if target in ['javascript', 'java', 'cpp']:
                    result.append(' ' * indent + '}')
                elif target == 'ruby':
                    result.append(' ' * indent + 'end')
                return result

        # Generate for loop header
        if target == 'python':
            result.append(' ' * indent + f'for {var} in range({start_val}, {end_val}):')
        elif target == 'javascript':
            result.append(' ' * indent + f'for (let {var} = {start_val}; {var} < {end_val}; {var}++) {{')
        elif target == 'java':
            result.append(' ' * indent + f'for (int {var} = {start_val}; {var} < {end_val}; {var}++) {{')
        elif target == 'cpp':
            result.append(' ' * indent + f'for (int {var} = {start_val}; {var} < {end_val}; {var}++) {{')
        elif target == 'ruby':
            result.append(' ' * indent + f'({start_val}...{end_val}).each do |{var}|')
        
        # Process body
        for line in lines[1:]:
            if line.strip():
                converted = self._convert_code(line.strip(), source, target)
                result.append(' ' * (indent + 4) + converted.strip())
        
        # Close
        if target in ['javascript', 'java', 'cpp']:
            result.append(' ' * indent + '}')
        elif target == 'ruby':
            result.append(' ' * indent + 'end')
        
        return result
    
    def _convert_while_loop(self, lines: List[str], source: str, target: str) -> List[str]:
        """Convert while loop"""
        first_line = lines[0].strip()
        result = []
        indent = len(lines[0]) - len(lines[0].lstrip())
        
        # Extract condition
        condition = 'true'
        if source == 'python':
            match = re.match(r'while\s+(.+):', first_line)
            if match:
                condition = match.group(1)
        elif source in ['javascript', 'java', 'cpp']:
            match = re.match(r'while\s*\((.+)\)', first_line)
            if match:
                condition = match.group(1)
        elif source == 'ruby':
            match = re.match(r'while\s+(.+)', first_line)
            if match:
                condition = match.group(1)
        
        # Generate
        if target == 'python':
            result.append(' ' * indent + f'while {condition}:')
        elif target in ['javascript', 'java', 'cpp']:
            result.append(' ' * indent + f'while ({condition}) {{')
        elif target == 'ruby':
            result.append(' ' * indent + f'while {condition}')
        
        # Body
        for line in lines[1:]:
            if line.strip():
                converted = self._convert_code(line.strip(), source, target)
                result.append(' ' * (indent + 4) + converted.strip())
        
        # Close
        if target in ['javascript', 'java', 'cpp']:
            result.append(' ' * indent + '}')
        elif target == 'ruby':
            result.append(' ' * indent + 'end')
        
        return result
    
    def _convert_if(self, lines: List[str], source: str, target: str) -> List[str]:
        """Convert if statement"""
        result = []
        indent = len(lines[0]) - len(lines[0].lstrip())
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            line_indent = len(line) - len(line.lstrip())
            
            if source == 'python':
                if stripped.startswith('if '):
                    condition = stripped[3:].rstrip(':')
                    if target == 'python':
                        result.append(' ' * indent + f'if {condition}:')
                    elif target in ['javascript', 'java', 'cpp']:
                        result.append(' ' * indent + f'if ({condition}) {{')
                    elif target == 'ruby':
                        result.append(' ' * indent + f'if {condition}')
                elif stripped.startswith('elif '):
                    condition = stripped[5:].rstrip(':')
                    if target == 'python':
                        result.append(' ' * indent + f'elif {condition}:')
                    elif target in ['javascript', 'java', 'cpp']:
                        result.append(' ' * indent + f'}} else if ({condition}) {{')
                    elif target == 'ruby':
                        result.append(' ' * indent + f'elsif {condition}')
                elif stripped == 'else:':
                    if target == 'python':
                        result.append(' ' * indent + 'else:')
                    elif target in ['javascript', 'java', 'cpp']:
                        result.append(' ' * indent + '} else {')
                    elif target == 'ruby':
                        result.append(' ' * indent + 'else')
                else:
                    # Body line
                    converted = self._convert_code(stripped, source, target)
                    result.append(' ' * (indent + 4) + converted.strip())
            else:
                if not stripped or stripped in ['{', '}']:
                    continue

                if re.match(r'^if\s*\(', stripped):
                    cond_match = re.match(r'^if\s*\((.*)\)\s*\{?$', stripped)
                    condition = cond_match.group(1).strip() if cond_match else 'true'
                    if target == 'python':
                        result.append(' ' * indent + f'if {condition}:')
                    elif target in ['javascript', 'java', 'cpp']:
                        result.append(' ' * indent + f'if ({condition}) {{')
                    elif target == 'ruby':
                        result.append(' ' * indent + f'if {condition}')
                    continue

                if re.match(r'^else\s+if\s*\(', stripped):
                    cond_match = re.match(r'^else\s+if\s*\((.*)\)\s*\{?$', stripped)
                    condition = cond_match.group(1).strip() if cond_match else 'true'
                    if target == 'python':
                        result.append(' ' * indent + f'elif {condition}:')
                    elif target in ['javascript', 'java', 'cpp']:
                        result.append(' ' * indent + f'}} else if ({condition}) {{')
                    elif target == 'ruby':
                        result.append(' ' * indent + f'elsif {condition}')
                    continue

                if re.match(r'^else\b', stripped):
                    if target == 'python':
                        result.append(' ' * indent + 'else:')
                    elif target in ['javascript', 'java', 'cpp']:
                        result.append(' ' * indent + '} else {')
                    elif target == 'ruby':
                        result.append(' ' * indent + 'else')
                    continue

                body_line = self._convert_generic_line(stripped, source, target, indent + 4)
                result.append(body_line)

        # Close
        if target in ['javascript', 'java', 'cpp']:
            result.append(' ' * indent + '}')
        elif target == 'ruby':
            result.append(' ' * indent + 'end')
        
        return result
    
    def _convert_print(self, line: str, source: str, target: str, indent: int) -> str:
        """Convert print statement"""
        content = ''
        
        if source == 'python':
            match = re.search(r'print\s*\((.*)\)', line)
            content = match.group(1) if match else ''
        elif source == 'javascript':
            match = re.search(r'console\.log\s*\((.*)\)', line)
            content = match.group(1) if match else ''
        elif source in ['java']:
            match = re.search(r'System\.out\.println?\s*\((.*)\)', line)
            content = match.group(1) if match else ''
        elif source == 'cpp':
            match = re.search(r'cout\s*<<\s*(.+?)\s*(?:<<\s*endl)?', line)
            content = match.group(1) if match else ''
        elif source == 'ruby':
            match = re.search(r'puts\s+(.+)', line)
            content = match.group(1) if match else ''
        
        content = content.strip().rstrip(';')
        
        if target == 'python':
            return ' ' * indent + f'print({content})'
        elif target == 'javascript':
            return ' ' * indent + f'console.log({content});'
        elif target == 'java':
            return ' ' * indent + f'System.out.println({content});'
        elif target == 'cpp':
            return ' ' * indent + f'cout << {content} << endl;'
        elif target == 'ruby':
            return ' ' * indent + f'puts {content}'
        
        return line
    
    def _convert_assignment(self, line: str, source: str, target: str, indent: int) -> str:
        """Convert variable assignment"""
        # Extract name and value
        name, value = '', ''
        
        if source == 'python':
            match = re.match(r'([a-zA-Z_]\w*)\s*=\s*(.+)', line)
            if match:
                name, value = match.groups()
        elif source == 'javascript':
            match = re.match(r'(?:let|const|var)\s+([a-zA-Z_]\w*)\s*=\s*(.+)', line)
            if match:
                name, value = match.groups()
        elif source in ['java', 'cpp']:
            match = re.match(r'(?:\w+)\s+([a-zA-Z_]\w*)\s*=\s*(.+)', line)
            if match:
                name, value = match.groups()
        elif source == 'ruby':
            match = re.match(r'([a-zA-Z_]\w*)\s*=\s*(.+)', line)
            if match:
                name, value = match.groups()
        
        value = value.strip().rstrip(';')
        
        if target == 'python':
            return ' ' * indent + f'{name} = {value}'
        elif target == 'javascript':
            return ' ' * indent + f'let {name} = {value};'
        elif target == 'java':
            return ' ' * indent + f'var {name} = {value};'
        elif target == 'cpp':
            return ' ' * indent + f'auto {name} = {value};'
        elif target == 'ruby':
            return ' ' * indent + f'{name} = {value}'
        
        return line
    
    def _convert_array_op(self, line: str, source: str, target: str, indent: int) -> str:
        """Convert array operations"""
        result = line
        
        # append/push
        if source == 'python' and '.append(' in line:
            match = re.search(r'(\w+)\.append\((.+)\)', line)
            if match:
                arr, val = match.groups()
                if target == 'javascript':
                    result = f'{arr}.push({val});'
                elif target == 'java':
                    result = f'{arr}.add({val});'
                elif target == 'cpp':
                    result = f'{arr}.push_back({val});'
                elif target == 'ruby':
                    result = f'{arr}.push({val})'
        
        # sort
        if source == 'python' and '.sort()' in line:
            match = re.search(r'(\w+)\.sort\(\)', line)
            if match:
                arr = match.group(1)
                if target == 'javascript':
                    result = f'{arr}.sort();'
                elif target == 'cpp':
                    result = f'sort({arr}.begin(), {arr}.end());'
                elif target == 'ruby':
                    result = f'{arr}.sort!'
        
        return ' ' * indent + result
    
    def _convert_return(self, line: str, source: str, target: str, indent: int) -> str:
        """Convert return statement"""
        match = re.search(r'return\s*(.+)', line)
        value = match.group(1) if match else ''
        value = value.strip().rstrip(';')
        
        if target in ['javascript', 'java', 'cpp']:
            return ' ' * indent + f'return {value};'
        elif target in ['python', 'ruby']:
            return ' ' * indent + f'return {value}'
        
        return line
    
    def _convert_generic_line(self, line: str, source: str, target: str, indent: int) -> str:
        """Convert generic line of code"""
        result = line
        
        # Handle list/array initialization
        if source == 'python' and '[' in line and ']' in line and '=' in line:
            # Python list to other languages
            match = re.match(r'(\w+)\s*=\s*\[(.+)\]', line)
            if match:
                name, values = match.groups()
                if target == 'javascript':
                    result = f'let {name} = [{values}];'
                elif target == 'java':
                    result = f'int[] {name} = {{{values}}};'
                elif target == 'cpp':
                    result = f'vector<int> {name} = {{{values}}};'
                elif target == 'ruby':
                    result = f'{name} = [{values}]'
        
        # Add semicolons for C-style languages
        if target in ['javascript', 'java', 'cpp']:
            if not result.endswith(';') and not result.endswith('{') and not result.endswith('}'):
                result = result + ';'
        
        return ' ' * indent + result
