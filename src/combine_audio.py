import numpy as np
import os
import librosa
import pandas as pd
from tqdm import tqdm
import soundfile as sf

def combine_audio(fg_audio, bg_audio, fg_sr, bg_sr, multiplier = 1):
    """
    Combine foreground and background audio clips with a given  .

    Parameters:
    - fg_audio (numpy array): Foreground audio clip array containing a known bird song.
    - bg_audio (numpy array): Background audio clip array sampled from the location.
    - multiplier (float): Multiplier value between 0 and 1.

    Returns:
    - combined_audio (numpy array): Combined audio clip array.
    """

    #new_sr = max(fg_sr, bg_sr)
    new_sr = bg_sr # we now only use bg_sr

    """
    # resample lowest sr audio to be at the higher sample rate
    if fg_sr > bg_sr:
        bg_audio = librosa.resample(bg_audio, orig_sr = bg_sr, target_sr = fg_sr)

    elif bg_sr > fg_sr:
        fg_audio = librosa.resample(fg_audio, orig_sr = fg_sr, target_sr = bg_sr)
    """

    # ensure fg_sr is always equal to bg_sr
    if fg_sr != bg_sr:
        fg_audio = librosa.resample(fg_audio, orig_sr = fg_sr, target_sr = bg_sr)


    # Multiply the foreground audio clip by the multiplier
    scaled_fg_audio = fg_audio * multiplier


    # Add scaled_foreground_audio and background_audio_clip
    combined_audio = scaled_fg_audio + bg_audio

    return combined_audio, new_sr

def remove_extension(filename):
    # Split the filename at the "." and take only the first part
    filename_without_extension = filename.split(".")[0]
    return filename_without_extension

def combine_and_save(
    fg_clips,
    bg_clips,
    output_folder,
    site_name,
    multipliers=None
):
    """
    Combine foreground audio with each background audio clip and save to disk.

    Parameters:
    - fg_clips (dict): {filename: (audio_clip, sr, species_name)}
    - bg_clips (dict): {filename: (audio_clip, sr)}
    - output_folder (str): Path to save combined audio.
    - site_name (str): The site name to be used in the output metadata.
    - multipliers (2D np.ndarray or None): Multiplier for each fg/bg pair.

    Returns:
    - df (pd.DataFrame): Info about saved clips.
    """
    saved_clips_info = []

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Convert bg_clips and fg_clips to lists for index tracking
    bg_clips_list = list(bg_clips.items())
    fg_clips_list = list(fg_clips.items())

    # Get muiltiplier for each foregroud background clip combination
    for bg_idx, (bg_filename, (bg_audio, bg_sr)) in enumerate(bg_clips_list):
        for fg_idx, (fg_filename, (fg_audio, fg_sr, species_name)) in enumerate(fg_clips_list):
            multiplier = multipliers[bg_idx, fg_idx] if multipliers is not None else 1

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Output filename: bg_filename_fg_filename.wav
            combined_filename = remove_extension(bg_filename) + "_" + fg_filename
            combined_audio, new_sr = combine_audio(fg_audio, bg_audio, fg_sr, bg_sr, multiplier=multiplier)
            output_path = os.path.join(output_folder, combined_filename)
            sf.write(output_path, combined_audio, new_sr)

            original_fg_amplitude = -20
            scaled_fg_amplitude = original_fg_amplitude + 20 * np.log10(multiplier)

            saved_clip_info = {
                'bg_filename': bg_filename,
                'fg_filename': fg_filename,
                'species_name': species_name,
                'site_name': site_name,  # <-- Use the passed-in site_name
                'output_filename': combined_filename,
                'fg_multiplier': multiplier,
                'scaled_fg_amplitude': scaled_fg_amplitude
            }
            saved_clips_info.append(saved_clip_info)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(saved_clips_info)

    return df


def load_foreground_audio(fg_audio_path, species_names):
    fg_clips = {}
    for species_name in species_names:
        if not os.path.exists(fg_audio_path):
            continue
        for filename in os.listdir(fg_audio_path):
            if filename.endswith(".wav"):
                file_path = os.path.join(fg_audio_path, filename)
                audio, sr = librosa.load(file_path, sr=None)
                fg_clips[filename] = (audio, sr, species_name)
    return fg_clips

def load_background_audio(bg_audio_path, site_names):
    """
    Returns a dict: {(site_name, filename): (audio, sr)}
    """
    bg_clips = {}
    for site_name in site_names:
        site_folder = os.path.join(bg_audio_path, site_name)
        if not os.path.exists(site_folder):
            continue
        for filename in os.listdir(site_folder):
            if filename.endswith(".wav"):
                file_path = os.path.join(site_folder, filename)
                audio, sr = librosa.load(file_path, sr=None)
                bg_clips[(site_name, filename)] = (audio, sr)
    return bg_clips

def run_combine_audio(
    fg_audio_path,
    bg_audio_path,
    combined_audio_path,
    output_path,
    species_names=None,
    site_names=None,
    db_range=(-80, -20),
    original_fg_amplitude=-20
):
    """
    Main function to combine foreground and background audio and save metadata.
    Args:
        fg_audio_path (str): Path to foreground audio root.
        bg_audio_path (str): Path to background audio root.
        combined_audio_path (str): Path to save combined audio.
        output_path (str): Path to save metadata CSV.
        species_names (list, optional): List of species names. If None, inferred from fg_audio_path.
        site_names (list, optional): List of site names. If None, inferred from bg_audio_path.
        db_range (tuple): Range of dB values for mixing.
        original_fg_amplitude (float): Reference amplitude for scaling.
    Returns:
        pd.DataFrame: Metadata DataFrame.
    """

    os.makedirs(combined_audio_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # Get species names from folder names in foreground audio path if not provided
    if species_names is None:
        species_names = [name for name in os.listdir(fg_audio_path) if os.path.isdir(os.path.join(fg_audio_path, name))]
    # Get site names from folders in background audio path if not provided
    if site_names is None:
        site_names = [name for name in os.listdir(bg_audio_path) if os.path.isdir(os.path.join(bg_audio_path, name))]

    # Load foreground audio clips for all species
    fg_clips = {}
    for species_name in species_names:
        species_folder = os.path.join(fg_audio_path, species_name)
        if not os.path.isdir(species_folder):
            print(f"Warning: {species_folder} not found, skipping {species_name}")
            continue
        fg_clips.update(load_foreground_audio(species_folder, [species_name]))


    # Load background audio clips
    bg_clips = load_background_audio(bg_audio_path, site_names)

    bg_clip_items = list(bg_clips.items())
    fg_clip_items = list(fg_clips.items())

    num_bg_clips = len(bg_clip_items)
    num_fg_clips = len(fg_clip_items)

    # Sample dB values uniformly between db_range
    db_values = np.random.uniform(db_range[0], db_range[1], size=(num_bg_clips, num_fg_clips))
    multipliers = 10 ** ((db_values + abs(original_fg_amplitude)) / 20)

    dfs = []


    for bg_idx, ((site_name, bg_filename), (bg_audio, bg_sr)) in tqdm(enumerate(bg_clip_items), total=num_bg_clips, desc="Processing bg_clips"):
        for fg_idx, (fg_filename, (fg_audio, fg_sr, species_name)) in enumerate(fg_clip_items):
            single_fg_clip = {fg_filename: (fg_audio, fg_sr, species_name)}
            single_bg_clip = {bg_filename: (bg_audio, bg_sr)}
            combined_audio_folder = os.path.join(combined_audio_path, site_name, species_name)

            os.makedirs(combined_audio_folder, exist_ok=True)
            temp_df = combine_and_save(
                single_fg_clip,
                single_bg_clip,
                combined_audio_folder,
                site_name=site_name,
                multipliers=np.array([[multipliers[bg_idx, fg_idx]]])
            )
            temp_df['species_name'] = species_name  # Ensure column is present

            dfs.append(temp_df)

    df = pd.concat(dfs, ignore_index=True)

    # Save the DataFrame to a CSV file
    csv_output_path = os.path.join(output_path, "combined_audio_metadata.csv")
    df.to_csv(csv_output_path, index=False)

    # Print total number of combined clips saved
    print(f"Total combined audio clips saved: {len(df)}")

    # Print metadata confirmation
    print(f"Combined audio metadata saved to {csv_output_path}")

# Example usage:
# run_combine_audio(
#     fg_audio_path=fg_audio_path,
#     bg_audio_path=bg_audio_path,
#     combined_audio_path=combined_audio_path,
#     output_path=output_path,
#     species_names=["Meadow Pipit", "Common Cuckoo"]
# )