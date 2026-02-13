#!/usr/bin/env python3
"""
Generates a project structure overview with descriptions.
"""
import os
from pathlib import Path
from typing import Dict

# Configuration: Descriptions for specific paths
DESCRIPTIONS: Dict[str, str] = {
    ".github/workflows": "CI/CD pipelines (GitHub Actions)",
    "config.toml": "Main application configuration",
    "corpora.json": "Corpus inventory (legacy/local)",
    "data": "Data storage (raw, processed, evals)",
    "data/evals": "Evaluation datasets (Golden Cases)",
    "data/processed": "Generated artifacts (chunks, TOCs, registry)",
    "data/raw": "Raw input documents (HTML)",
    "data/sample_docs": "Text files for simple testing",
    "docs": "Documentation and inventory",
    "runs": "Local execution logs and debug outputs",
    "scripts": "CLI entry points and maintenance scripts",
    "src": "Source code",
    "src/common": "Shared utilities (settings, registry, schema)",
    "src/engine": "Core RAG runtime (retrieval, planning, routing)",
    "src/ingestion": "ETL pipelines (EUR-Lex, HTML chunking)",
    "src/services": "Application layer (Ask API, Eval suite)",
    "src/main.py": "Main CLI entry point",
    "tests": "Test suite (Pytest)",
    "tools": "Internal tooling (inventory, debug)",
    "ui": "Frontend application",
    "ui/streamlit_app.py": "Streamlit entry point",
    ".env": "Environment variables (secrets)",
    "requirements.txt": "Python dependencies",
}

# Directories to ignore in the tree
IGNORE_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    ".DS_Store",
    "data/vector_store", # Too large/binary
    "ui/.cache",
}

def _should_ignore(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    parts = rel.parts
    
    # Check exact match or parent match in IGNORE_DIRS
    for part in parts:
        if part in IGNORE_DIRS:
            return True
        if part.startswith("__"): # __pycache__
            return True
            
    # Check specific full paths
    if str(rel) in IGNORE_DIRS:
        return True
        
    return False

def print_tree(root: Path, prefix: str = ""):
    # Get all children
    try:
        children = sorted([p for p in root.iterdir()], key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return

    # Filter ignored
    filtered = []
    repo_root = Path(__file__).resolve().parents[1]
    for p in children:
        if not _should_ignore(p, repo_root):
            filtered.append(p)
            
    total = len(filtered)
    for i, path in enumerate(filtered):
        is_last = (i == total - 1)
        connector = "└── " if is_last else "├── "
        
        rel_path = str(path.relative_to(repo_root))
        desc = DESCRIPTIONS.get(rel_path, "")
        
        # Print the node
        line = f"{prefix}{connector}{path.name}"
        if desc:
            # Align descriptions
            padding = max(0, 60 - len(line))
            line = f"{line}{' ' * padding} # {desc}"
        print(line)
        
        # Recurse if directory
        if path.is_dir():
            extension = "    " if is_last else "│   "
            print_tree(path, prefix + extension)

def main():
    root = Path(__file__).resolve().parents[1]
    print("PROJECT STRUCTURE OVERVIEW")
    print(f"Root: {root}")
    print("\nTREE")
    print(".")
    print_tree(root)
    
    print("\nLEGEND")
    print("- src/engine:    Core logic (RAG, Retrieval, Planning)")
    print("- src/ingestion: Data processing (ETL, Chunking)")
    print("- src/common:    Shared utilities (Settings, Schema)")
    print("- src/services:  Business logic (API, Evals)")

if __name__ == "__main__":
    main()
