import os
import json
import numpy as np
import librosa
import scipy.signal as signal
import pandas as pd

def compute_bandpass_rms(file_path, min_freq, max_freq):
    """Compute RMS after applying a bandpass filter to an audio file."""
    audio, sr = librosa.load(file_path, sr=None)
    sos = signal.butter(N=4, Wn=[min_freq, max_freq], btype='bandpass', fs=sr, output='sos')
    filtered_audio = signal.sosfilt(sos, audio)
    rms = np.sqrt(np.mean(filtered_audio**2))
    return rms

def weighted_avg_amplitude(vocalization_amplitude, vocalization_time):
    """Compute weighted average amplitude in dB."""
    # Convert amplitude to linear scale
    linear_amplitudes = [10 ** (amp / 10) for amp in vocalization_amplitude]

    # Calculate the weighted average in linear scale
    weighted_sum = sum(a * t for a, t in zip(linear_amplitudes, vocalization_time))
    total_time = sum(vocalization_time)

    # Avoid division by zero if total_time is zero
    avg_linear_amplitude = weighted_sum / total_time if total_time > 0 else 0

    # Convert back to decibels
    return 10 * np.log10(avg_linear_amplitude) if avg_linear_amplitude > 0 else -np.inf

def parse_songmeter_bg_filename(filename):
    """
    Extract datetime from Songmeter Micro filename.
    Returns dict with 'bg_datetime' or empty dict if parsing fails.
    """
    try:
        parts = filename.split("_")
        date_str = parts[1]
        time_str = parts[2].split(".")[0]
        from datetime import datetime
        bg_datetime = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        return {'bg_datetime': bg_datetime}
    except Exception:
        return {}

def compute_bg_rms_for_fg_bg_combos(vocalization_df, bg_audio_folder):
    """
    For each foreground vocalization and each background file (recursively found in subfolders),
    compute bandpass RMS using the min/max freq for that foreground.
    Returns a DataFrame with columns:
    ['fg_filename', 'bg_filename', 'site_name', 'min_freq', 'max_freq', 'background_amplitude', 'bg_datetime']
    """
    results = []
    # Recursively find all .wav files and their site_name
    bg_files = []
    for root, dirs, files in os.walk(bg_audio_folder):
        for f in files:
            if f.lower().endswith('.wav'):
                site_name = os.path.basename(root)
                bg_files.append((os.path.join(root, f), f, site_name))

    for _, fg_row in vocalization_df.iterrows():
        fg_filename = fg_row['filename']
        min_freq = fg_row['min_freq']
        max_freq = fg_row['max_freq']

        for bg_path, bg_filename, site_name in bg_files:
            try:
                rms = compute_bandpass_rms(bg_path, min_freq, max_freq)
            except Exception as e:
                print(f"Error processing {bg_filename} for {fg_filename}: {e}")
                rms = np.nan

            # Convert linear RMS to dB
            rms_dB = 20 * np.log10(rms)

            # Parse datetime from background filename
            bg_datetime = parse_songmeter_bg_filename(bg_filename).get('bg_datetime', None)

            results.append({
                'fg_filename': fg_filename,
                'bg_filename': bg_filename,
                'site_name': site_name,
                'min_freq': min_freq,
                'max_freq': max_freq,
                'background_amplitude': rms_dB,
                'bg_datetime': bg_datetime
            })

    return pd.DataFrame(results)

def run_background_amplitude_analysis(
    species_names,
    bg_audio_folder,
    output_csv_path,
    project_root="."
):
    """
    Compute background bandpass RMS for all fg/bg combos for each species and save to CSV.

    Args:
        species_names (list): List of species names (strings).
        bg_audio_folder (str): Path to background audio folder (with site subfolders).
        output_csv_path (str): Path to save output CSV.
        project_root (str): Path to project root (default: current directory).
    Returns:
        pd.DataFrame: The resulting DataFrame.
    """
    all_results = []
    for species_name in species_names:
        # Build path to vocalizations.json for this species
        vocalization_json = os.path.join(
            project_root, "data", "foreground_audio", species_name, "vocalizations.json"
        )
        if not os.path.exists(vocalization_json):
            print(f"Warning: {vocalization_json} not found, skipping {species_name}")
            continue

        with open(vocalization_json, "r") as f:
            data = json.load(f)
        vocalization_df = pd.DataFrame(data)
        vocalization_df['species_name'] = species_name  # Add species column

        # Compute weighted avg amplitude, SNR, total vocalization time
        vocalization_df['avg_vocalization_amplitude'] = vocalization_df.apply(
            lambda row: weighted_avg_amplitude(row['vocalization_amplitude'], row['vocalization_time']), axis=1
        )
        vocalization_df['SNR'] = vocalization_df['avg_vocalization_amplitude'] - vocalization_df['bg_amplitude']
        vocalization_df['total_vocalization_time'] = vocalization_df['vocalization_time'].apply(sum)

        # Compute background RMS for each fg/bg combo using fg min/max freq
        bg_rms_df = compute_bg_rms_for_fg_bg_combos(vocalization_df, bg_audio_folder)
        bg_rms_df['species_name'] = species_name  # Add species column to output

        all_results.append(bg_rms_df)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)

        # Extract the output folder path
        output_folder = os.path.dirname(output_csv_path)

        # Create the folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        final_df.to_csv(output_csv_path, index=False)
        print(f"Background amplitudes saved to: {output_csv_path}")    
    else:
        print("No results to save.")

if __name__ == "__main__":
    # Example usage for CLI/testing
    vocalization_json = os.path.join(r"data\foreground_audio\Meadow Pipit\vocalizations.json")
    bg_audio_folder = r"data\background_audio"
    output_csv_path = r"data\output\background_amplitudes.csv"

    run_background_amplitude_analysis(
        vocalization_json,
        bg_audio_folder,
        output_csv_path
    )