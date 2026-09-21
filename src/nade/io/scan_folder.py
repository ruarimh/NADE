#!/usr/bin/env python3
"""
Scan folder for audio files
Returns a list of audio files found in the specified directory
"""

import argparse
import json
import os
import glob
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = ['wav', 'mp3', 'flac', 'ogg', 'm4a']


def infer_site_name(folder_path, file_path):
    """Infer the site name from a file path relative to the scanned folder.

    Examples:
    - folder_path='/data/' and file_path='/data/site_1/clip.wav' -> 'site_1'
    - folder_path='/data/site_1/' and file_path='/data/site_1/clip.wav' -> 'site_1'
    - folder_path='/data/' and file_path='/data/clip.wav' -> 'data'
    """
    folder_path = os.path.normpath(folder_path)
    file_path = os.path.normpath(file_path)

    rel_path = os.path.relpath(file_path, folder_path)
    rel_parts = Path(rel_path).parts

    if len(rel_parts) >= 2:
        return rel_parts[0]

    return os.path.basename(folder_path) or os.path.basename(os.path.normpath(os.path.dirname(folder_path)))


def scan_folder(folder_path):
    """Scan folder recursively for audio files"""
    logger.info(f"Scanning folder: {folder_path}")
    audio_files = []
    
    for extension in AUDIO_EXTENSIONS:
        pattern = os.path.join(folder_path, '**', f'*.{extension}')
        files = glob.glob(pattern, recursive=True)
        audio_files.extend(files)
        
        if files:
            logger.info(f"Found {len(files)} .{extension} files")
    
    # Sort files alphabetically
    audio_files.sort()
    
    logger.info(f"Total audio files found: {len(audio_files)}")
    
    return audio_files


def scan_folder_with_metadata(folder_path):
    """Scan folder and return metadata dictionary.

    This function is safe to call from other Python modules and returns
    a dict similar to the JSON printed by the CLI `main()`.

    The metadata includes the inferred site_name for each file, using the
    first directory beneath the scanned root as the site identifier. For example,
    files under /data/site_1/ will carry site_name == 'site_1'.
    """
    audio_files = scan_folder(folder_path)
    file_metadata = []
    site_names = []

    for file_path in audio_files:
        site_name = infer_site_name(folder_path, file_path)
        file_metadata.append({
            'path': file_path,
            'site_name': site_name,
        })
        if site_name not in site_names:
            site_names.append(site_name)

    site_name = site_names[0] if len(site_names) == 1 else None

    result = {
        'count': len(audio_files),
        'folder': folder_path,
        'site_names': site_names,
        'file_metadata': file_metadata,
    }

    return result

def main():
    parser = argparse.ArgumentParser(description='Scan folder for audio files')
    parser.add_argument('folder', help='Folder path to scan')
    
    args = parser.parse_args()
    
    try:
        if not os.path.exists(args.folder):
            raise FileNotFoundError(f"Folder not found: {args.folder}")

        if not os.path.isdir(args.folder):
            raise ValueError(f"Path is not a directory: {args.folder}")

        result = scan_folder_with_metadata(args.folder)

        print(json.dumps(result))
        
    except Exception as e:
        logger.error(f"Error scanning folder: {e}")
        error_result = {
            'error': str(e),
            'files': [],
            'count': 0,
            'folder': args.folder if 'args' in locals() else ''
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()