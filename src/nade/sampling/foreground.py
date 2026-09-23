import json
import logging
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

# Compute average vocalization amplitude in linear scale, weighted by vocalization time
def weighted_avg_amplitude(vocalization_amplitude, vocalization_time):
    # Convert amplitude to linear scale
    linear_amplitudes = [10 ** (amp / 10) for amp in vocalization_amplitude]
    
    # Calculate the weighted average in linear scale
    weighted_sum = sum(a * t for a, t in zip(linear_amplitudes, vocalization_time))
    total_time = sum(vocalization_time)
    
    # Avoid division by zero if total_time is zero
    avg_linear_amplitude = weighted_sum / total_time if total_time > 0 else 0
    
    # Convert back to decibels
    return 10 * np.log10(avg_linear_amplitude) if avg_linear_amplitude > 0 else -np.inf


logger = logging.getLogger(__name__)

# Matches vocalisations.json, vocalizations.json, vocalisation.json, vocalization.json,
# plus prefixed variants such as "<species>_vocalizations.json".
_VOCALISATION_JSON_RE = re.compile(r"vocali[sz]ations?\.json$", re.IGNORECASE)

AUDIO_EXTENSIONS = ['wav', 'mp3', 'flac', 'ogg', 'm4a']


def _wav_name(filename: str) -> str:
    """Swap a known audio extension for .wav (or append .wav if there isn't one)."""
    path = Path(filename)
    if path.suffix.lower().lstrip(".") in AUDIO_EXTENSIONS:
        return path.with_suffix(".wav").name
    return path.name + ".wav"


def _find_vocalisation_json(species_dir: Path) -> Path | None:
    """Return the vocalisation metadata JSON inside species_dir, or None if there isn't one."""
    matches = sorted(
        (p for p in species_dir.iterdir()
         if p.is_file() and _VOCALISATION_JSON_RE.search(p.name)),
        # Bare names like "vocalisations.json" win over prefixed ones.
        key=lambda p: (not p.name.lower().startswith("vocali"), p.name),
    )
    if len(matches) > 1:
        logger.warning("%s: multiple vocalisation JSON files found, using %s",
                       species_dir.name, matches[0].name)
    return matches[0] if matches else None


def _source_pcm_subtype(audio_path: Path) -> str | None:
    """Keep the source bit depth (e.g. PCM_24) for wav/flac inputs; otherwise use the wav default."""
    try:
        info = sf.info(audio_path)
    except Exception:
        return None  # e.g. m4a, which libsndfile can't open
    return info.subtype if info.format in ("WAV", "FLAC") and info.subtype.startswith("PCM") else None


def normalise_foreground_vocalisations(
        foreground_audio_path: str | Path,
        target_amplitude: float = -20.0) -> Path:
    """Normalise each foreground vocalisation clip to a target average amplitude.

    Expects the layout:
        <foreground_audio_path>/<species_name>/<clip>.<wav|mp3|flac|ogg|m4a>
        <foreground_audio_path>/<species_name>/vocalisations.json   (or vocalizations.json, etc.)

    For every entry in the JSON, the weighted average vocalisation amplitude (dB) is
    computed, a gain is applied so that it equals `target_amplitude`, and the clip is
    written to:
        <foreground_audio_path>_normalised/<species_name>/<clip>.wav

    Input can be any format librosa/soundfile can decode (m4a needs ffmpeg installed);
    output is always wav, with the original sample rate and channel count. Clips that
    are missing, unreadable, have no valid amplitude, or appear more than once in the
    JSON are skipped with a warning. If a gain would push a clip beyond full scale, it
    is written as 32-bit float so nothing is clipped.

    Returns the root output directory.
    """
    foreground_audio_path = Path(foreground_audio_path).resolve()
    output_root = foreground_audio_path.with_name(foreground_audio_path.name + "_normalised")

    for species_dir in sorted(p for p in foreground_audio_path.iterdir() if p.is_dir()):
        json_path = _find_vocalisation_json(species_dir)
        if json_path is None:
            logger.warning("%s: no vocalisation JSON found, skipping species", species_dir.name)
            continue

        with open(json_path, encoding="utf-8") as f:
            vocalisations = json.load(f)

        output_dir = output_root / species_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        seen = set()
        for voc in vocalisations:
            filename = voc["filename"]
            out_name = _wav_name(filename)

            if out_name in seen:
                logger.warning("%s: %s listed more than once, skipping duplicate",
                               species_dir.name, filename)
                continue
            seen.add(out_name)

            audio_path = species_dir / filename
            if not audio_path.is_file():
                logger.warning("%s: audio file %s not found, skipping", species_dir.name, filename)
                continue

            avg_amplitude = weighted_avg_amplitude(
                voc["vocalization_amplitude"], voc["vocalization_time"])
            if not np.isfinite(avg_amplitude):
                logger.warning("%s: no valid amplitude for %s, skipping", species_dir.name, filename)
                continue

            gain_db = target_amplitude - avg_amplitude
            gain = 10 ** (gain_db / 20)  # dB -> linear

            try:
                # sr=None keeps the native sample rate; mono=False keeps all channels.
                audio, sr = librosa.load(audio_path, sr=None, mono=False)
                if audio.ndim == 2:
                    audio = audio.T  # librosa: (channels, samples) -> soundfile: (samples, channels)
                adjusted_audio = audio * gain

                subtype = _source_pcm_subtype(audio_path)
                peak = float(np.max(np.abs(adjusted_audio))) if adjusted_audio.size else 0.0
                if peak > 1.0:
                    logger.warning("%s: %s would clip after %+.1f dB gain (peak %.2f), writing as float",
                                   species_dir.name, filename, gain_db, peak)
                    subtype = "FLOAT"

                sf.write(output_dir / out_name, adjusted_audio, sr, subtype=subtype)
            except Exception as e:
                logger.warning("%s: failed to process %s (%s), skipping", species_dir.name, filename, e)

    return output_root