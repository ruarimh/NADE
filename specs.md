# NADE package — implementation spec

Status legend: `[existing]` = port/refactor of current code, `[new]` = new feature

---

## 1. Proposed layout

```
NADE/
├── data/
│   ├── foreground_audio/
│   ├── background_audio/
│   ├── combined_audio/
│   └── output/
├── src/
│   └── nade/
│       ├── __init__.py
│       ├── io/
│       │   ├── __init__.py
│       │   ├── audio.py              # load/save clips, resampling, mono-down-mixing
│       │   └── filename_parsers.py   # [existing→generalized] recorder-brand datetime parsing
│       ├── amplitude.py              # [existing] band-limited RMS / dBFS measurement
│       ├── sampling/
│       │   ├── __init__.py
│       │   ├── background.py         # [existing: random] + [new: near-detection]
│       │   └── foreground.py         # [existing: uniform-dB] + [new: detection-informed]
│       ├── combine.py                # [existing] fg+bg -> combined clips + metadata
│       ├── labeling.py               # [new] suggest high-confidence clips to verify
│       ├── aggregate.py              # [new/cleanup] aggregate by site / time / both
│       ├── config.py                 # run configuration (the "few selections" step)
│       └── pipeline.py               # run_nade(config) — the "hit play" entrypoint
├── tests/
│   ├── reference_data/               # small slice already processed by the OLD pipeline
│   ├── test_amplitude.py
│   ├── test_combine.py
│   └── test_pipeline_parity.py       # end-to-end: new output vs reference_data, within tolerance
├── notebooks/
│   └── NADE_demo.ipynb
├── figures/
├── README.md
├── pyproject.toml
└── requirements.yml
```

---

## 2. `io/filename_parsers.py` — recorder-agnostic datetime extraction `[existing→generalized]`

Replaces the hardcoded `parse_songmeter_bg_filename()`. Goal: user picks a recorder type
(or we auto-detect it), and we extract `(datetime, site_name | None)` from any filename.

```python
@dataclass
class ParsedFilename:
    datetime: datetime
    site_name: str | None = None

class FilenameParser(Protocol):
    name: str  # e.g. "songmeter_micro", "audiomoth"

    def matches(self, filename: str) -> bool:
        """True if this parser's pattern matches the filename. Used for auto-detection."""

    def parse(self, filename: str) -> ParsedFilename:
        """Raises ValueError if filename doesn't match this parser's pattern."""
```

- `SongmeterMicroParser` — current behavior, pattern `SMM08941_20230422_103000.wav`.
- `AudioMothParser` — **flagged open decision below**, confirm exact naming convention(s) in use.
- `register_parser(parser: FilenameParser) -> None` — lets a user add a custom recorder.
- `get_parser(name: str) -> FilenameParser`
- `parse_datetime_from_filename(filename: str, recorder_type: str | None = None) -> ParsedFilename`
  - if `recorder_type` given, use that parser directly
  - else try each registered parser's `.matches()`; raise a clear, actionable error
    (not a silent fallback) if zero or more than one match

**Test:** every filename in `data/` (both foreground and background) round-trips through
this and produces the same datetimes the old `parse_songmeter_bg_filename()` did.

---

## 3. `amplitude.py` — band-limited amplitude measurement `[existing]`

```python
def measure_band_amplitude(
    audio: np.ndarray, sr: int, min_freq: float, max_freq: float
) -> float:
    """Band-pass filter to [min_freq, max_freq], return RMS amplitude in dBFS.
    Filter design: <see Open decisions — must match original for parity>.
    """

def run_background_amplitude_analysis(
    combined_metadata: pd.DataFrame,
    background_dir: Path,
    ...
) -> pd.DataFrame:
    """Returns columns exactly as in current README:
    ['fg_filename','bg_filename','site_name','min_freq','max_freq',
     'background_amplitude','bg_datetime']
    """
```

**Test:** re-run against `tests/reference_data/`, compare `background_amplitude` column
to the old CSV output within tolerance (see §9).

---

## 4. `sampling/background.py` — background clip selection

```python
@dataclass
class BackgroundClip:
    site_name: str
    file_path: Path
    start_s: float
    end_s: float
    datetime: datetime
    source: Literal["random", "near_detection"]

def sample_background_random(  # [existing]
    site_dir: Path,
    n_samples: int,
    clip_duration_s: float,
    time_window: tuple[time, time] = (time(4, 0), time(21, 0)),
    recorder_type: str | None = None,
    seed: int | None = None,
) -> list[BackgroundClip]:
    """Uniform random sampling across survey duration within time_window, as currently done."""

def sample_background_near_detections(  # [new]
    site_dir: Path,
    detections: pd.DataFrame,   # needs columns: file_path, detection_start_s, detection_end_s
    window_before_s: float,
    window_after_s: float,
    exclusion_buffer_s: float,   # gap kept clear around the vocalization itself
    n_per_detection: int = 1,
    recorder_type: str | None = None,
    seed: int | None = None,
) -> list[BackgroundClip]:
    """For each detection, sample background clip(s) temporally near it.
    Edge cases handled per Open decisions §10 (file boundaries, overlapping windows,
    clustered detections).
    """
```

---

## 5. `sampling/foreground.py` — foreground amplitude selection

```python
def select_foreground_amplitudes_uniform(  # [existing]
    min_dbfs: float, max_dbfs: float, n: int, seed: int | None = None
) -> list[float]:
    """e.g. -80 to -20 dBFS as you currently use."""

def select_foreground_amplitudes_from_detections(  # [new]
    detections: pd.DataFrame,        # needs: file_path, start_s, end_s, species
    vocal_freq_ranges: dict[str, tuple[float, float]],  # species -> (min_freq, max_freq)
) -> pd.DataFrame:
    """For every detection, measure the amplitude of its source clip within the
    species' vocalizing frequency range (via amplitude.measure_band_amplitude),
    and use that as the target foreground amplitude for combination.
    Returns a df with one row per detection: [file_path, species, target_dbfs].
    """
```

---

## 6. `combine_audio.py` — foreground/background combination `[existing]`

```python
def run_combine_audio(
    foreground_dir: Path,
    vocalizations_json: Path,     # per data/foreground_audio/<species>/vocalizations.json
    background_dir: Path,
    amplitude_mode: Literal["uniform", "detection_informed"],
    output_dir: Path,
    uniform_range_dbfs: tuple[float, float] | None = None,   # required if mode == "uniform"
    detections: pd.DataFrame | None = None,                   # required if mode == "detection_informed"
) -> pd.DataFrame:
    """Combines each foreground vocalization with each background clip at the chosen
    amplitude, writes combined .wav files (or whatever the input filetype is) to output_dir, and returns metadata:
    one row per combined clip (fg source, bg source, site, datetime, target amplitude,
    output filename, species_name, ...).
    """
```

**Test:** with `amplitude_mode="uniform"` and the same seed/range as the old code,
combined clip metadata + a spot-check of actual audio samples should match
`tests/reference_data/` (allowing for float rounding — see §9).

---
## 7. Model inference

We do not provide any code for this -- users likely have their own model setup in their own way, some will use GUI, some will use command line, there are many different models... so, instead of forcing them to use models within this package for inference, we just ask the user to run the model on the generated combined audio clips.

---

## 8. `labeling.py`, `aggregate.py`

```python
def suggest_vocalizations_to_verify(  # [new]
    detections: pd.DataFrame, confidence_threshold: float = 0.98, n: int | None = None
) -> pd.DataFrame:
    """Filters/ranks real-world (not combined) detections above threshold as
    candidates for manual verification. A suggestion tool only — does not replace
    manual annotation."""

def aggregate_by_site(df: pd.DataFrame) -> pd.DataFrame: ...      # [new/cleanup]
def aggregate_by_time(df: pd.DataFrame, freq: str = "1D") -> pd.DataFrame: ...
def aggregate_by_site_and_time(df: pd.DataFrame, freq: str = "1D") -> pd.DataFrame: ...
```

---

## 9. `config.py` + `pipeline.py` — the "hit play" experience

```python
@dataclass
class NadeConfig:
    foreground_dir: Path
    background_dir: Path
    vocalizations_json: Path
    recorder_type: str | None          # None = auto-detect
    amplitude_mode: Literal["uniform", "detection_informed"]
    uniform_range_dbfs: tuple[float, float] | None
    detector_name: str                 # "birdnet" | "perch" | user-registered name
    detector_kwargs: dict
    aggregation: Literal["site", "time", "both"]

    @classmethod
    def from_yaml(cls, path: Path) -> "NadeConfig": ...

def run_nade(config: NadeConfig) -> "NadeResults":
    """Single entrypoint: sample background -> combine audio -> let user run inference
    -> aggregate. This is the function a non-coding user effectively calls."""
```

---

## 10. Open decisions — fill these in before/while implementing

These are the assumptions that won't show up as bugs — they'll show up as silently
wrong numbers. Worth pinning down explicitly rather than letting an agent guess:

- [ ] **dBFS convention**: what's full-scale referenced to (peak amplitude of the bit
      depth? something else?) — must match your existing code exactly.
- [ ] **Band-pass filter design** used in `measure_band_amplitude` (filter type, order,
      zero-phase/`filtfilt` or not) — needs to match the original for parity.
- [ ] **Vocalizing frequency range**: fixed per species, or derived per-clip from
      `vocalizations.json` min/max?
- [ ] **Near-detection sampling**: exact window sizes, exclusion buffer around the
      vocalization, behavior when the window crosses a file boundary, behavior when
      detections cluster close together (dedupe? allow overlapping samples?).
- [ ] **`suggest_vocalizations_to_verify`**: global threshold or per-species? capped
      at `n` total or `n` per species? tie-breaking rule?
- [ ] **Aggregation time bucketing default** (hourly? daily?) and how gaps/missing
      recording periods are handled.
- [ ] **AudioMoth (and any other recorder) filename convention(s)** you actually need
      supported.

---

## 11. Validation

For every module above, before moving to the next: run against `tests/reference_data/`
(a slice of data already processed by the current pipeline) and compare outputs.
Exact bit-for-bit equality is an unrealistic bar across a refactor — define a tolerance
instead, e.g.:

- Amplitude/dBFS values: agree within some `atol` (e.g. 1e-4) — pick a number and put it here.
- Combined-clip metadata: exact match on categorical fields (site, filenames), tolerance
  on numeric ones.
- Aggregated confidence tables: agree within tolerance after aggregation — this is your
  real acceptance bar, since it's what downstream analysis depends on.

`test_pipeline_parity.py` should run the *whole* `run_nade()` path end-to-end and check
against a known-good aggregated output, not just unit-test individual functions in isolation.