/**
 * CodeBridge - Simple LeetCode Style Interface
 */

const API_URL = 'http://localhost:5000';

// Language modes for CodeMirror
const MODES = {
    python: 'python',
    javascript: 'javascript',
    java: 'text/x-java',
    cpp: 'text/x-c++src',
    ruby: 'ruby'
};

// Default code templates
const TEMPLATES = {
    python: `# Python Code
print("Hello, World!")
`,
    javascript: `// JavaScript Code
console.log("Hello, World!");
`,
    java: `// Java Code
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
`,
    cpp: `// C++ Code
#include <iostream>
using namespace std;

int main() {
    cout << "Hello, World!" << endl;
    return 0;
}
`,
    ruby: `# Ruby Code
puts "Hello, World!"
`
};

// Global variables
let codeEditor, convertedEditor;
let currentLanguage = 'python';
let targetLanguage = 'javascript';

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initEditors();
    initEvents();
    initResizablePanels();
});

function initEditors() {
    // Main code editor
    codeEditor = CodeMirror.fromTextArea(document.getElementById('code-editor'), {
        mode: MODES[currentLanguage],
        lineNumbers: true,
        tabSize: 4,
        indentUnit: 4,
        lineWrapping: true
    });
    
    // Set default template
    codeEditor.setValue(TEMPLATES[currentLanguage]);
    
    // Converted code editor
    convertedEditor = CodeMirror.fromTextArea(document.getElementById('converted-editor'), {
        mode: MODES[targetLanguage],
        lineNumbers: true,
        tabSize: 4,
        lineWrapping: true
    });
}

function initEvents() {
    // Language selection change
    document.getElementById('language-select').addEventListener('change', function(e) {
        currentLanguage = e.target.value;
        codeEditor.setOption('mode', MODES[currentLanguage]);
    });
    
    // Run button (main editor)
    document.getElementById('run-btn').addEventListener('click', runCode);
    
    // Clear button (main editor)
    document.getElementById('clear-btn').addEventListener('click', clearAll);
    
    // Convert button
    document.getElementById('convert-btn').addEventListener('click', convertCode);
    
    // Target language change
    document.getElementById('target-language').addEventListener('change', function(e) {
        targetLanguage = e.target.value;
        document.getElementById('target-label').textContent = e.target.options[e.target.selectedIndex].text;
    });
    
    // Run converted code button
    document.getElementById('run-converted-btn').addEventListener('click', runConvertedCode);
    
    // Clear converted code button
    document.getElementById('clear-converted-btn').addEventListener('click', clearConverted);
    
    // Keyboard shortcut: Ctrl+Enter to run
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            // Run the active editor's code
            if (document.activeElement.closest('.left-panel')) {
                runCode();
            } else {
                runConvertedCode();
            }
        }
    });
}

function initResizablePanels() {
    const mainContent = document.querySelector('.main-content');
    const leftPanel = document.getElementById('left-panel');
    const rightPanel = document.getElementById('right-panel');
    const verticalResizer = document.getElementById('vertical-resizer');
    const leftHorizontalResizer = document.getElementById('left-horizontal-resizer');
    const rightHorizontalResizer = document.getElementById('right-horizontal-resizer');
    const leftEditorSection = document.getElementById('left-editor-section');
    const leftBottomSection = document.getElementById('left-bottom-section');
    const rightEditorSection = document.getElementById('right-editor-section');
    const rightOutputSection = document.getElementById('right-output-section');

    if (!mainContent || !leftPanel || !rightPanel || !verticalResizer) {
        return;
    }

    const MIN_PANEL_WIDTH = 320;
    const MIN_SECTION_HEIGHT = 110;
    let dragState = null;

    const refreshEditors = () => {
        if (codeEditor) {
            codeEditor.refresh();
        }
        if (convertedEditor) {
            convertedEditor.refresh();
        }
    };

    const startVerticalDrag = (event) => {
        if (window.innerWidth <= 900) {
            return;
        }
        event.preventDefault();
        const mainRect = mainContent.getBoundingClientRect();
        const leftRect = leftPanel.getBoundingClientRect();
        const resizerRect = verticalResizer.getBoundingClientRect();

        dragState = {
            type: 'vertical',
            startX: event.clientX,
            leftWidth: leftRect.width,
            containerWidth: mainRect.width,
            resizerWidth: resizerRect.width
        };

        document.body.classList.add('is-resizing', 'is-resizing-vertical');
        window.addEventListener('pointermove', onDrag);
        window.addEventListener('pointerup', stopDrag);
    };

    const startHorizontalDrag = (event, topElement, bottomElement) => {
        event.preventDefault();
        const topRect = topElement.getBoundingClientRect();
        const bottomRect = bottomElement.getBoundingClientRect();

        dragState = {
            type: 'horizontal',
            startY: event.clientY,
            topHeight: topRect.height,
            bottomHeight: bottomRect.height,
            topElement,
            bottomElement
        };

        document.body.classList.add('is-resizing', 'is-resizing-horizontal');
        window.addEventListener('pointermove', onDrag);
        window.addEventListener('pointerup', stopDrag);
    };

    const onDrag = (event) => {
        if (!dragState) {
            return;
        }

        if (dragState.type === 'vertical') {
            const dx = event.clientX - dragState.startX;
            const maxLeftWidth = dragState.containerWidth - MIN_PANEL_WIDTH - dragState.resizerWidth;
            const nextLeftWidth = Math.max(MIN_PANEL_WIDTH, Math.min(maxLeftWidth, dragState.leftWidth + dx));
            leftPanel.style.flex = `0 0 ${nextLeftWidth}px`;
            rightPanel.style.flex = '1 1 auto';
        } else {
            const dy = event.clientY - dragState.startY;
            const totalHeight = dragState.topHeight + dragState.bottomHeight;
            const nextTopHeight = Math.max(
                MIN_SECTION_HEIGHT,
                Math.min(totalHeight - MIN_SECTION_HEIGHT, dragState.topHeight + dy)
            );
            const nextBottomHeight = totalHeight - nextTopHeight;

            dragState.topElement.style.flex = `0 0 ${nextTopHeight}px`;
            dragState.bottomElement.style.flex = `0 0 ${nextBottomHeight}px`;
        }

        refreshEditors();
    };

    const stopDrag = () => {
        dragState = null;
        document.body.classList.remove('is-resizing', 'is-resizing-vertical', 'is-resizing-horizontal');
        window.removeEventListener('pointermove', onDrag);
        window.removeEventListener('pointerup', stopDrag);
        refreshEditors();
    };

    verticalResizer.addEventListener('pointerdown', startVerticalDrag);
    if (leftHorizontalResizer && leftEditorSection && leftBottomSection) {
        leftHorizontalResizer.addEventListener('pointerdown', (event) => {
            startHorizontalDrag(event, leftEditorSection, leftBottomSection);
        });
    }
    if (rightHorizontalResizer && rightEditorSection && rightOutputSection) {
        rightHorizontalResizer.addEventListener('pointerdown', (event) => {
            startHorizontalDrag(event, rightEditorSection, rightOutputSection);
        });
    }

    window.addEventListener('resize', refreshEditors);
}

async function runCode() {
    const code = codeEditor.getValue();
    const stdin = document.getElementById('stdin-input').value;
    const outputEl = document.getElementById('output-text');
    
    if (!code.trim()) {
        outputEl.textContent = 'Please enter some code.';
        outputEl.className = 'output-error';
        return;
    }
    
    outputEl.textContent = 'Running...';
    outputEl.className = '';
    
    try {
        const response = await fetch(`${API_URL}/api/execute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                code: code,
                language: currentLanguage,
                stdin: stdin
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            const result = data.result;
            if (result.success) {
                outputEl.textContent = result.stdout || '(No output)';
                outputEl.className = 'output-success';
            } else {
                outputEl.textContent = result.stderr || 'Execution failed';
                outputEl.className = 'output-error';
            }
        } else {
            outputEl.textContent = 'Error: ' + data.message;
            outputEl.className = 'output-error';
        }
    } catch (error) {
        outputEl.textContent = 'Error: Cannot connect to server.\nMake sure backend is running on port 5000.';
        outputEl.className = 'output-error';
    }
}

async function runConvertedCode() {
    const code = convertedEditor.getValue();
    const outputEl = document.getElementById('converted-output-text');
    
    if (!code.trim()) {
        outputEl.textContent = 'No converted code to run.';
        outputEl.className = 'output-error';
        return;
    }
    
    outputEl.textContent = 'Running...';
    outputEl.className = '';
    
    try {
        const response = await fetch(`${API_URL}/api/execute`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                code: code,
                language: targetLanguage,
                stdin: ''
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            const result = data.result;
            if (result.success) {
                outputEl.textContent = result.stdout || '(No output)';
                outputEl.className = 'output-success';
            } else {
                outputEl.textContent = result.stderr || 'Execution failed';
                outputEl.className = 'output-error';
            }
        } else {
            outputEl.textContent = 'Error: ' + data.message;
            outputEl.className = 'output-error';
        }
    } catch (error) {
        outputEl.textContent = 'Error: Cannot connect to server.';
        outputEl.className = 'output-error';
    }
}

async function convertCode() {
    const code = codeEditor.getValue();
    const outputEl = document.getElementById('output-text');
    
    if (!code.trim()) {
        outputEl.textContent = 'Please enter code to convert.';
        outputEl.className = 'output-error';
        return;
    }
    
    outputEl.textContent = 'Converting...';
    outputEl.className = '';
    
    try {
        const response = await fetch(`${API_URL}/api/convert`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                code: code,
                source_language: currentLanguage,
                target_language: targetLanguage
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            convertedEditor.setOption('mode', MODES[targetLanguage]);
            convertedEditor.setValue(data.result.converted_code);
            outputEl.textContent = 'Conversion complete!';
            outputEl.className = 'output-success';
        } else {
            outputEl.textContent = 'Error: ' + data.message;
            outputEl.className = 'output-error';
        }
    } catch (error) {
        outputEl.textContent = 'Error: Cannot connect to server.';
        outputEl.className = 'output-error';
    }
}

function clearAll() {
    codeEditor.setValue(TEMPLATES[currentLanguage]);
    document.getElementById('stdin-input').value = '';
    document.getElementById('output-text').textContent = '';
    document.getElementById('output-text').className = '';
}

function clearConverted() {
    convertedEditor.setValue('');
    document.getElementById('converted-output-text').textContent = '';
    document.getElementById('converted-output-text').className = '';
}
