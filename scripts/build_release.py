#!/usr/bin/env python3
"""
Release Packaging Script for Vertex Lock Blender Add-on / Extension.
Packages the modular add-on into a clean .zip distribution archive without
bytecode, cache files, or development artifacts.
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT_DIR / "blender_vertex_lock"
DIST_DIR = ROOT_DIR / "dist"

EXCLUDE_PATTERNS = {
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".DS_Store",
    "Thumbs.db",
    "ehthumbs.db",
    "*.swp",
    "*~",
}

def should_exclude(path: Path) -> bool:
    for part in path.parts:
        if part == "__pycache__":
            return True
    for pattern in EXCLUDE_PATTERNS:
        if path.match(pattern):
            return True
    return False

def build_package():
    if not SOURCE_DIR.exists():
        print(f"[ERROR] Source directory does not exist: {SOURCE_DIR}")
        sys.exit(1)

    DIST_DIR.mkdir(parents=True, exist_ok=True)

    # Read version from manifest
    version = "1.1.0"
    manifest_file = SOURCE_DIR / "blender_manifest.toml"
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("version"):
                    version = line.split("=")[1].strip().strip('"').strip("'")
                    break

    zip_filename = f"blender_vertex_lock-v{version}.zip"
    generic_zip_filename = "blender_vertex_lock.zip"
    zip_path = DIST_DIR / zip_filename
    generic_zip_path = DIST_DIR / generic_zip_filename

    print(f"Building Vertex Lock release package v{version}...")
    print(f"Source: {SOURCE_DIR}")
    print(f"Target: {zip_path}")

    # Create zip archive containing blender_vertex_lock folder
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(SOURCE_DIR):
            # Avoid descending into __pycache__
            dirs[:] = [d for d in dirs if d != "__pycache__"]

            for file in files:
                file_path = Path(root) / file
                if should_exclude(file_path):
                    continue

                rel_to_source = file_path.relative_to(SOURCE_DIR)
                # Store inside the blender_vertex_lock folder in the zip
                archive_name = Path("blender_vertex_lock") / rel_to_source
                zf.write(file_path, str(archive_name))
                print(f"  + Added: {archive_name}")

    # Create copy with generic name for convenient download linking
    shutil.copy2(zip_path, generic_zip_path)

    # Also copy standalone single-file vertex_lock.py to dist if present
    single_file = ROOT_DIR / "vertex_lock.py"
    if single_file.exists():
        shutil.copy2(single_file, DIST_DIR / "vertex_lock.py")
        print(f"  + Copied standalone: dist/vertex_lock.py")

    size_kb = zip_path.stat().st_size / 1024
    print("-" * 50)
    print(f"[SUCCESS] Built: {zip_path.name} ({size_kb:.1f} KB)")
    print(f"[SUCCESS] Built: {generic_zip_filename}")
    print(f"Artifacts located in: {DIST_DIR}")

if __name__ == "__main__":
    build_package()
