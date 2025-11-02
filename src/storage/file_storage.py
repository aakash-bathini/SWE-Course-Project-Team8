"""
File storage utilities for model uploads and downloads
Handles local file system storage with integrity checks
"""

import os
import hashlib
import shutil
import zipfile
import logging
from pathlib import Path
from typing import Optional, Dict, Any, BinaryIO

logger = logging.getLogger(__name__)

# Storage configuration
STORAGE_ROOT = os.path.join(os.getcwd(), "uploads")
CHUNK_SIZE = 8192  # 8KB chunks for streaming


def ensure_storage_directory() -> None:
    """Create storage directory if it doesn't exist"""
    os.makedirs(STORAGE_ROOT, exist_ok=True)


def get_artifact_directory(artifact_id: str) -> str:
    """Get the directory path for an artifact"""
    return os.path.join(STORAGE_ROOT, artifact_id)


def save_uploaded_file(artifact_id: str, file_content: bytes, filename: str) -> Dict[str, Any]:
    """
    Save uploaded file to storage
    Returns dict with path, size, and checksum
    """
    try:
        ensure_storage_directory()
        
        # Create artifact directory
        artifact_dir = get_artifact_directory(artifact_id)
        os.makedirs(artifact_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(artifact_dir, filename)
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Calculate SHA256 checksum
        checksum = calculate_checksum(file_path)
        file_size = os.path.getsize(file_path)
        
        logger.info(f"Saved file {filename} for artifact {artifact_id}: "
                   f"{file_size} bytes, checksum {checksum}")
        
        return {
            "path": file_path,
            "filename": filename,
            "size": file_size,
            "checksum": checksum
        }
        
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise


def calculate_checksum(file_path: str) -> str:
    """Calculate SHA256 checksum of a file"""
    sha256 = hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(CHUNK_SIZE)
            if not data:
                break
            sha256.update(data)
    
    return sha256.hexdigest()


def extract_zip(zip_path: str, extract_to: str) -> list[str]:
    """
    Extract ZIP file and return list of extracted files
    """
    try:
        extracted_files = []
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Validate ZIP
            if zip_ref.testzip() is not None:
                raise ValueError("ZIP file is corrupted")
            
            # Extract all files
            zip_ref.extractall(extract_to)
            extracted_files = zip_ref.namelist()
        
        logger.info(f"Extracted {len(extracted_files)} files from ZIP")
        return extracted_files
        
    except zipfile.BadZipFile:
        logger.error("Invalid ZIP file")
        raise ValueError("Invalid ZIP file format")
    except Exception as e:
        logger.error(f"Error extracting ZIP: {e}")
        raise


def find_model_card(artifact_dir: str) -> Optional[str]:
    """
    Find README.md or model card in artifact directory
    Returns path to model card if found
    """
    try:
        # Common model card filenames
        card_names = ["README.md", "readme.md", "README", "MODEL_CARD.md"]
        
        for card_name in card_names:
            card_path = os.path.join(artifact_dir, card_name)
            if os.path.exists(card_path):
                return card_path
        
        # Check in subdirectories (one level deep)
        for item in os.listdir(artifact_dir):
            item_path = os.path.join(artifact_dir, item)
            if os.path.isdir(item_path):
                for card_name in card_names:
                    card_path = os.path.join(item_path, card_name)
                    if os.path.exists(card_path):
                        return card_path
        
        return None
        
    except Exception as e:
        logger.error(f"Error finding model card: {e}")
        return None


def read_model_card(card_path: str) -> str:
    """Read and return model card content"""
    try:
        with open(card_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading model card: {e}")
        return ""


def filter_files_by_aspect(artifact_dir: str, aspect: str) -> list[str]:
    """
    Filter files in artifact directory by aspect
    aspect: 'full', 'weights', 'datasets', 'code'
    """
    try:
        all_files = []
        
        # Walk through directory
        for root, dirs, files in os.walk(artifact_dir):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
        
        if aspect == "full":
            return all_files
        
        # Filter by file type
        filtered_files = []
        
        for file_path in all_files:
            filename = os.path.basename(file_path).lower()
            
            if aspect == "weights":
                # Common model weight file extensions
                if any(filename.endswith(ext) for ext in 
                      ['.pt', '.pth', '.bin', '.safetensors', '.h5', '.ckpt', '.weights']):
                    filtered_files.append(file_path)
            
            elif aspect == "datasets":
                # Common dataset file extensions
                if any(filename.endswith(ext) for ext in
                      ['.csv', '.json', '.jsonl', '.txt', '.parquet', '.arrow']):
                    filtered_files.append(file_path)
            
            elif aspect == "code":
                # Code files
                if any(filename.endswith(ext) for ext in
                      ['.py', '.ipynb', '.sh', '.yaml', '.yml', '.toml']):
                    filtered_files.append(file_path)
        
        return filtered_files
        
    except Exception as e:
        logger.error(f"Error filtering files by aspect: {e}")
        return []


def create_zip_from_files(files: list[str], artifact_dir: str, output_path: str) -> str:
    """
    Create ZIP file from list of files
    Returns path to created ZIP
    """
    try:
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files:
                # Calculate relative path for ZIP
                arcname = os.path.relpath(file_path, artifact_dir)
                zipf.write(file_path, arcname)
        
        logger.info(f"Created ZIP with {len(files)} files at {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating ZIP: {e}")
        raise


def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get file metadata"""
    try:
        stat = os.stat(file_path)
        return {
            "path": file_path,
            "size": stat.st_size,
            "checksum": calculate_checksum(file_path),
            "filename": os.path.basename(file_path)
        }
    except Exception as e:
        logger.error(f"Error getting file info: {e}")
        raise


def delete_artifact_files(artifact_id: str) -> bool:
    """Delete all files for an artifact"""
    try:
        artifact_dir = get_artifact_directory(artifact_id)
        
        if os.path.exists(artifact_dir):
            shutil.rmtree(artifact_dir)
            logger.info(f"Deleted files for artifact {artifact_id}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error deleting artifact files: {e}")
        return False
