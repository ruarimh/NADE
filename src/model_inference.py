import os
import json
import pandas as pd
from birdnetlib.analyzer import Analyzer
from birdnetlib import Recording
from birdnetlib.batch import DirectoryMultiProcessingAnalyzer
from contextlib import contextmanager
import time
from pathlib import Path
import multiprocessing


class DirectoryAnalyzer(DirectoryMultiProcessingAnalyzer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Calculate the total size of the .wav files in the directory
        self.total_size_gb, self.num_files = self.calculate_total_wav_size()  
        self.total_size_gb = self.total_size_gb / (1024 ** 3) # Convert bytes to GB
        self.estimated_time_seconds = self.estimate_runtime(self.total_size_gb, self.processes)

        print(f"Analyzer initialized with {self.processes} processes.")
        print(f"Number of .wav files: {self.num_files}") 
        print(f"Total size of .wav files: {self.total_size_gb:.2f} GB")
        print(f"Estimated processing time: {self.estimated_time_seconds // 60:.0f} minutes {self.estimated_time_seconds % 60:.0f} seconds")

    @contextmanager
    def suppress_stdout(self):
        """
        A context manager to suppress both stdout and stderr.
        """
        with open(os.devnull, 'w') as devnull:
            old_stdout = os.dup(1)
            old_stderr = os.dup(2)
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            try:
                yield
            finally:
                os.dup2(old_stdout, 1)
                os.dup2(old_stderr, 2)
                os.close(old_stdout)
                os.close(old_stderr)

    def process(self):
        start_time = time.time()

        with self.suppress_stdout():
            super().process()
        
        time_taken = time.time() - start_time
        minutes, seconds = divmod(time_taken, 60)
        print(f"Time taken: {int(minutes)} minutes and {int(seconds)} seconds")


    def calculate_total_wav_size(self):
        # Calculate the total size of .wav files in the directory
        wav_files = list(Path(self.directory).rglob("*.wav"))  # Convert to list to get count
        num_files = len(wav_files)
        total_size = sum(f.stat().st_size for f in wav_files)
        return total_size, num_files


    def estimate_runtime(self, total_size_gb, processes):
        # Benchmark: 15 GB took 18 minutes and 37 seconds (1117 seconds) using 22 processes
        benchmark_size_gb = 1.85
        benchmark_time_seconds = 6 * 60 + 54  # 18 minutes and 37 seconds
        benchmark_processes = 22
        
        # Scale the estimated time based on the number of processes
        estimated_time_seconds = (total_size_gb / benchmark_size_gb) * benchmark_time_seconds
        process_scaling_factor = benchmark_processes / processes
        estimated_time_seconds *= process_scaling_factor
        
        return estimated_time_seconds


    def save_files(self):
        print("Processing complete!")


        count = 0

        print(f"Saving {len(self.directory_recordings)} .json files...")
        print()
        print("The first 3 .json files are below:")

        for recording in self.directory_recordings:
            # Remove 'common_name', 'label', and 'end_time' fields from each detection
            simplified_detections = []
            for detection in recording.detections:
                simplified_detection = {
                    'scientific_name': detection['scientific_name'],
                    'start_time': detection['start_time'],
                    'confidence': detection['confidence']
                }
                simplified_detections.append(simplified_detection)

            simplified_detections.sort(key=lambda x: x['confidence'], reverse=True)

            # Create output JSON filename by replacing the .wav extension
            json_filename = os.path.splitext(recording.path)[0] + '.json'

            # Save the simplified detections to a .json file
            with open(json_filename, 'w') as json_file:
                json.dump(simplified_detections, json_file, indent=4)
            
            if count < 3:
                print(f"Saved detections to {json_filename}")

                count += 1

def run_birdnet_on_combined_audio(
    combined_audio_root,
    species_names,
    analyzer,
    is_common_name=True,
    lookup_table=None,
    min_conf=0.01,
    skip_processing=False
):
    """
    Run BirdNET on all combined audio files and return a DataFrame with confidence values.

    Parameters:
    - combined_audio_root (str): Root folder containing site/species/combined.wav files.
    - species_names (list): List of species to process.
    - analyzer (Analyzer): BirdNET Analyzer object.
    - is_common_name (bool): Whether to use common names for matching.
    - lookup_table (dict): Lookup table for name conversion.
    - min_conf (float): Minimum confidence threshold.
    - skip_processing (bool): If True, skip BirdNET processing and only parse existing JSON.

    Returns:
    - pd.DataFrame: DataFrame with columns ['output_filename', 'confidence', 'site_name', 'species_name']
    """
    results = []

    for site_name in os.listdir(combined_audio_root):
        site_path = os.path.join(combined_audio_root, site_name)
        if not os.path.isdir(site_path):
            continue
        for species_name in species_names:
            species_path = os.path.join(site_path, species_name)
            if not os.path.isdir(species_path):
                continue

            # Optionally run BirdNET batch processing here if needed
            if not skip_processing:
                analyzer.directory = species_path
                analyzer.directory_recordings = []
                analyzer.process()
                analyzer.save_files()

            # Parse JSON results for each .wav file
            for file in os.listdir(species_path):
                if file.endswith('.json'):
                    json_path = os.path.join(species_path, file)
                    with open(json_path, 'r') as f:
                        detections = json.load(f)

                    confidence = 0.0
                    for entry in detections:
                        if is_common_name:
                            # Convert scientific to common name if lookup_table is provided
                            if lookup_table:
                                common_name = convert_species(
                                    entry['scientific_name'],
                                    lookup_table,
                                    "scientific_name",
                                    "EB_english_name"
                                )
                            else:
                                common_name = entry.get('common_name', None)
                            if common_name == species_name:
                                confidence = entry['confidence']
                                break
                        else:
                            if entry['scientific_name'] == species_name:
                                confidence = entry['confidence']
                                break

                    output_filename = file.replace('.json', '.wav')
                    results.append({
                        'output_filename': output_filename,
                        'confidence': confidence,
                        'site_name': site_name,
                        'species_name': species_name
                    })

    return pd.DataFrame(results)

def convert_species(species_name, lookup_table, from_field, to_field):
    for species_info in lookup_table.values():
        if species_info.get(from_field) == species_name:
            return species_info.get(to_field)
    return None



def run_model_inference(
    combined_audio_root,
    species_names,
    lookup_table_path,
    output_csv_path,
    n_cpus=None,
    min_conf=0.01,
    skip_processing=False
):
    """
    Run BirdNET inference on combined audio and save results to CSV.

    Args:
        combined_audio_root (str): Path to combined audio root folder.
        species_names (list): List of species names to process.
        lookup_table_path (str): Path to species lookup JSON.
        output_csv_path (str): Path to save output CSV.
        n_cpus (int, optional): Number of CPUs to use. Defaults to all available minus one.
        min_conf (float, optional): Minimum confidence threshold.
        skip_processing (bool, optional): If True, only parse existing JSONs.

    Returns:
        pd.DataFrame: The resulting DataFrame.
    """
    if n_cpus is None:
        n_cpus = max(1, multiprocessing.cpu_count() - 1)

    combined_audio_folders = [
        os.path.join(combined_audio_root, d)
        for d in os.listdir(combined_audio_root)
        if os.path.isdir(os.path.join(combined_audio_root, d))
    ]

    analyzer = DirectoryAnalyzer(
        directory=combined_audio_folders[0],  # Will be updated per site/species
        processes=n_cpus,
        min_conf=min_conf
    )

    with open(lookup_table_path, 'r') as json_file:
        lookup_table = json.load(json_file)

    df = run_birdnet_on_combined_audio(
        combined_audio_root,
        species_names,
        analyzer,
        is_common_name=True,
        lookup_table=lookup_table,
        min_conf=min_conf,
        skip_processing=skip_processing
    )

    df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    # CLI/test usage
    run_model_inference(
        combined_audio_root=r"data\combined_audio",
        species_names=["Meadow Pipit"],
        lookup_table_path=r"data\species_lookup.json",
        output_csv_path=r"data\output\model_confidences.csv",
        n_cpus=None,
        min_conf=0.01,
        skip_processing=False
    )