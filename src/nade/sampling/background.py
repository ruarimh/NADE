"""Background sampling utilities.

This module provides `sample_background_audio()` which takes a metadata
dictionary describing raw audio files and writes out sampled background
clips plus a CSV describing them.

The metadata dict is expected to have the structure::

    result = {
        'count': len(audio_files),
        'folder': folder_path,
        'site_names': site_names,
        'file_metadata': file_metadata,
    }

where each entry in `file_metadata` is a dict with at least::

    {'path': '/absolute/or/relative/path.wav', 'site_name': 'site_1'}

The function does not modify the original/raw audio files.
"""

from pathlib import Path
import math
import random
import csv
import os
from typing import Dict, List, Optional

import librosa
import soundfile as sf


def sample_background_audio(
    metadata: Dict,
    model_input_length: float = 3.0,
    background_fraction: float = 0.01,
    sampling_type: str = "random",
    output_folder: str = "data/background_audio",
    seed: Optional[int] = None,
    overwrite: bool = False,
):
    """Sample background audio from raw files described in `metadata`.

    Args:
        metadata: dictionary describing files (see module docstring)
        model_input_length: length of each sample in seconds
        background_fraction: fraction of all possible segments to sample
        sampling_type: one of 'random' or 'detection_based' (stub)
        output_folder: where to write sampled audio and CSV. If None,
            a `background_audio` folder next to `metadata['folder']` is used.
        seed: RNG seed for reproducibility
        overwrite: if True, allow overwriting existing output files/CSV

    Returns:
        path to the CSV file created (str)
    """

    if seed is not None:
        random.seed(seed)

    file_metadata: List[Dict] = metadata.get("file_metadata", [])
    if not file_metadata:
        raise ValueError("metadata contains no 'file_metadata' entries")

    # Determine output folder
    base_folder = Path(output_folder)
    base_folder.mkdir(parents=True, exist_ok=True)

    # Build list of candidate segments (path, site_name, start_time)
    candidates = []
    for entry in file_metadata:
        path = Path(entry["path"])
        site = entry.get("site_name", entry.get("site", "unknown_site"))
        if not path.exists():
            # try relative to metadata folder
            maybe = Path(metadata.get("folder", "")) / path
            if maybe.exists():
                path = maybe
            else:
                # skip missing files but warn
                print(f"warning: file not found, skipping: {path}")
                continue

        # compute duration in seconds using librosa (uses soundfile backend)
        try:
            duration = librosa.get_duration(filename=str(path))
        except Exception:
            print(f"warning: could not read duration for {path}, skipping")
            continue

        n_segments = math.floor(duration / float(model_input_length))
        if n_segments <= 0:
            continue

        for i in range(n_segments):
            start_time = float(i) * float(model_input_length)
            candidates.append({"path": str(path), "site": site, "start_time": start_time})

    total = len(candidates)
    if total == 0:
        raise ValueError("no eligible segments found for the given model_input_length")

    n_to_sample = int(round(total * float(background_fraction)))
    if n_to_sample <= 0:
        n_to_sample = 1
    if n_to_sample > total:
        n_to_sample = total

    if sampling_type == "random":
        chosen = random.sample(candidates, n_to_sample)
    elif sampling_type == "detection_based":
        # detection_based sampling will require per-site CSVs and extra logic.
        # Here we provide a small stub that simply selects a fraction of
        # candidates per site. Keep structure so extending is straightforward.
        chosen = []
        by_site = {}
        for c in candidates:
            by_site.setdefault(c["site"], []).append(c)
        for site, items in by_site.items():
            m = max(1, int(round(len(items) * float(background_fraction))))
            chosen.extend(random.sample(items, min(m, len(items))))
        # if we undershot target, pad with random picks
        if len(chosen) < n_to_sample:
            remaining = [c for c in candidates if c not in chosen]
            add = min(n_to_sample - len(chosen), len(remaining))
            if add > 0:
                chosen.extend(random.sample(remaining, add))
    else:
        raise ValueError(f"unknown sampling_type: {sampling_type}")

    # Group chosen by file so we load each raw file at most once
    chosen_by_file = {}
    for c in chosen:
        chosen_by_file.setdefault(c["path"], []).append(c)

    csv_rows = []
    for file_path, segments in chosen_by_file.items():
        audio, sr = librosa.load(file_path, sr=None)
        for seg in segments:
            start_s = seg["start_time"]
            start_sample = int(round(start_s * sr))
            length_samples = int(round(model_input_length * sr))
            end_sample = start_sample + length_samples
            clip = audio[start_sample:end_sample]

            site_dir = base_folder / seg.get("site", "unknown_site")
            site_dir.mkdir(parents=True, exist_ok=True)

            orig = Path(file_path)
            out_name = f"{orig.stem}_{int(start_s)}s_{int(start_s + model_input_length)}s.wav"
            out_path = site_dir / out_name
            if out_path.exists() and not overwrite:
                print(f"skipping existing file {out_path}")
            else:
                sf.write(str(out_path), clip, sr)

            csv_rows.append({
                "original_file": str(orig),
                "site": seg.get("site", ""),
                "start_time": start_s,
                "end_time": start_s + model_input_length,
                "saved_path": str(out_path),
            })

    # write CSV summary
    csv_path = base_folder / "background_samples.csv"
    if csv_path.exists() and not overwrite:
        print(f"CSV already exists at {csv_path}, not overwriting")
    else:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["original_file", "site", "start_time", "end_time", "saved_path"])
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)

    return str(csv_path)