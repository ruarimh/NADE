"""
detection_loader
================

Load the detection / classification output of acoustic models into ONE standard
pandas ``DataFrame``, regardless of which tool produced it.

Standard columns (always present, in this order)
-------------------------------------------------
filename         recording the detection belongs to (as written by the tool, or inferred from
                 the name of the detections file when the tool does not write it)
start_time       seconds from the start of the recording
end_time         seconds from the start of the recording (NaN if the tool gives no end/duration)
common_name      common (vernacular) name, when it is known / can be derived
scientific_name  scientific (Latin) name, when it is known / can be derived
confidence       model confidence as written by the tool (probability, score, logit, ...)
label            raw class label as written by the tool (e.g. "Turdus merula_Eurasian Blackbird",
                 "MAWR", "Corvus corax"); NaN when the tool wrote separate name columns instead

Formats recognised out of the box (``available_formats()``)
-----------------------------------------------------------
BirdNET-Analyzer   BirdNET_CombinedTable.csv / <rec>.BirdNET.results.csv (+ .parquet),
                   BirdNET_SelectionTable.txt / <rec>.BirdNET.selection.table.txt (Raven),
                   BirdNET_Kaleidoscope.csv / <rec>.BirdNET.results.kaleidoscope.csv,
                   BirdNET_AudacityLabels.txt / <rec>.BirdNET.results.txt (Audacity), old "R table" csv
HawkEars           scores.csv / rarities.csv, <rec>_scores.txt (Audacity, "LABEL;score"),
                   <rec>.HawkEars.selection.table.txt (Raven)
BirdCODE (ESP)     BirdCODE_predictions/<rec>.txt  (Begin Time (s), End Time (s), Species, Score)
Perch              chirp inference csv (filename, timestamp_s, label, logit),
                   perch-hoplite agile csv (idx, project, filename, window_start, window_end, label, logits)
Raven Pro          *.selections.txt selection tables (multi-file tables via "File Offset (s)",
                   duplicate rows per View removed)
Kaleidoscope       id.csv / cluster.csv style tables (INDIR, FOLDER, IN FILE, OFFSET, DURATION, ...)
Audacity           any headerless "start<TAB>end<TAB>label[<TAB>confidence]" label track
Wide tables        OpenSoundscape style: file,start_time,end_time,<one column per class>
Generic tables     any csv / tsv / txt / parquet / feather / json / jsonl / xlsx whose header contains
                   recognisable column names (see ``COLUMN_SYNONYMS``)

Typical use
-----------
    from detection_loader import find_detections_file, load_detections

    path = find_detections_file("/runs/site_A/birdnet_output")   # best detections file in the folder
    df = load_detections(path)                                    # -> standard DataFrame

    # or, in one go, load *all* detection files of the best format found in the folder:
    df = load_detections("/runs/site_A/birdnet_output")

Extending
---------
Subclass ``TableFormat`` (header-based tables) or ``DetectionFormat`` (anything else) and
decorate it with ``@register_format``:

    @register_format
    class MyToolCSV(TableFormat):
        name = "mytool_csv"
        tool = "MyTool"
        required = frozenset({"clip", "onset_sec", "offset_sec", "species", "p"})
        base_score = 70
        filename_patterns = ((r"mytool_.*\\.csv$", 20),)
        column_map = {"clip": "filename", "onset_sec": "start_time", "offset_sec": "end_time",
                      "species": "label", "p": "confidence"}

Unknown-but-sensible files usually need no new class at all: the generic reader maps column
names through ``COLUMN_SYNONYMS`` (add entries there or pass ``column_map=`` to
``load_detections``).
"""

from __future__ import annotations

import codecs
import json
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "STANDARD_COLUMNS",
    "COLUMN_SYNONYMS",
    "AUDIO_EXTENSIONS",
    "find_detections_file",
    "find_detections_files",
    "load_detections",
    "describe_file",
    "detect_format",
    "available_formats",
    "register_format",
    "DetectionFormat",
    "TableFormat",
    "LoadOptions",
    "Sniff",
    "sniff_file",
    "split_label",
    "load_label_map",
    "normalise_column",
]

STANDARD_COLUMNS: list[str] = [
    "filename",
    "start_time",
    "end_time",
    "common_name",
    "scientific_name",
    "confidence",
    "label",
]

AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".wav", ".wave", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".wma",
     ".aif", ".aiff", ".w4v", ".wac", ".mp4", ".webm"}
)

# File extensions that may hold a detections table.
TABLE_EXTENSIONS: frozenset[str] = frozenset(
    {".csv", ".tsv", ".tab", ".txt", ".lbl", ".parquet", ".pq", ".feather",
     ".json", ".jsonl", ".ndjson", ".xlsx", ".xls"}
)
_STRUCTURED_EXTENSIONS = frozenset({".parquet", ".pq", ".feather", ".xlsx", ".xls",
                                    ".json", ".jsonl", ".ndjson"})

# Files that commonly sit next to detections but are not detections themselves.
IGNORED_FILENAMES: frozenset[str] = frozenset(
    {"birdnet.analyze-params.csv", "classes.csv", "recordings.csv", "details.csv",
     "species_list.txt", "labels.txt", "requirements.txt", "manifest.csv", "manifest.json",
     "exclude.txt", "include.txt", "readme.txt", "readme.md", "license.txt", "config.yaml"}
)

# ---------------------------------------------------------------------------------------------
# Column-name synonyms (all names are *normalised*: lower case, units in () removed,
# non-alphanumerics -> "_").  Order matters: the first synonym found in a table wins.
# ---------------------------------------------------------------------------------------------
COLUMN_SYNONYMS: dict[str, list[str]] = {
    "filename": [
        "filename", "file_name", "file", "filepath", "file_path", "path", "begin_path",
        "begin_file", "in_file", "recording", "recording_path", "recording_file", "recording_name",
        "recording_file_name", "original_file_name", "audio_file", "audio_path", "audio_filename",
        "audio", "source_file", "source", "clip", "clip_name", "input", "input_file", "wav",
        "wav_file", "wavfile", "sound_file", "soundfile", "media_file", "media", "original_wav",
    ],
    "start_time": [
        "start_time", "start_time_s", "start_time_sec", "start_time_seconds", "start", "start_s",
        "start_sec", "start_seconds", "begin_time", "begin_time_s", "begin", "begin_s", "onset",
        "onset_s", "onset_time", "timestamp_s", "time_start", "t_start", "tstart", "t0", "tmin",
        "min_t", "window_start", "clip_start", "segment_start", "offset", "offset_s",
        "start_offset", "file_offset", "from",
    ],
    "end_time": [
        "end_time", "end_time_s", "end_time_sec", "end_time_seconds", "end", "end_s", "end_sec",
        "end_seconds", "finish", "offset_time", "time_end", "t_end", "tend", "t1", "tmax", "max_t",
        "window_end", "clip_end", "segment_end", "stop", "stop_time", "end_offset", "to",
    ],
    "duration": [
        "duration", "duration_s", "duration_sec", "duration_seconds", "dur", "length", "length_s",
        "delta_time", "clip_duration", "segment_duration", "window_size", "window_s",
    ],
    "common_name": [
        "common_name", "commonname", "common", "english_name", "englishname", "com_name",
        "species_common_name", "vernacular_name", "vernacular", "primary_com_name", "comname", "birdnet_english_name"
    ],
    "scientific_name": [
        "scientific_name", "scientificname", "scientific", "sci_name", "sciname", "latin_name",
        "latin", "species_scientific_name", "binomial", "taxon", "taxon_name", "sci", "birdnet_scientific_name",
    ],
    "confidence": [
        "confidence", "conf", "confidence_score", "score", "scores", "probability", "prob",
        "probs", "p", "logit", "logits", "likelihood", "detection_score", "detection_confidence",
        "det_score", "class_prob", "det_prob", "certainty", "prediction_score", "pred_score",
        "activation", "top1dist", "match_ratio", "value",
    ],
    "label": [
        "label", "labels", "species", "species_name", "class", "class_name", "classname",
        "name", "annotation", "annotations", "category", "tag", "tags", "sound_type",
        "call_type", "top1match", "auto_id", "manual_id", "species_code", "code", "prediction",
        "predicted_label", "predicted_class", "predicted_species", "pred", "target",
        "event_label", "event", "detection", "type",
    ],
}

# Columns that must never be treated as class columns of a wide table.
_NON_CLASS_COLUMNS: frozenset[str] = frozenset(
    {"selection", "view", "channel", "low_freq", "high_freq", "lat", "latitude", "lon",
     "longitude", "week", "overlap", "sensitivity", "min_conf", "idx", "index", "unnamed_0",
     "level_0", "id", "project", "model", "species_list", "delta_time", "date", "time", "hour",
     "sample_rate", "sr", "n_channels"}
)

_SUFFIX_RE = re.compile(
    r"(\.birdnet\.selection\.table|\.birdnet\.results\.kaleidoscope|\.birdnet\.results"
    r"|\.hawkears\.selection\.table|_hawkears|_scores|\.table\.\d+\.selections|\.selections"
    r"|\.selection\.table|_predictions|\.predictions|_labels|\.labels|_detections|\.detections"
    r"|_birdnet|\.birdnet|_results|\.results)$",
    re.IGNORECASE,
)
_COMBINED_STEMS = {"birdnet_audacitylabels", "birdnet_combinedtable", "birdnet_selectiontable",
                   "birdnet_kaleidoscope", "birdnet_rtable", "scores", "rarities"}


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------
def normalise_column(name: Any) -> str:
    """'Begin Time (s)' -> 'begin_time', 'AUTO ID*' -> 'auto_id', ' timestamp_s' -> 'timestamp_s'."""
    s = str(name).strip().lower()
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


_norm = normalise_column


def _is_number(s: str) -> bool:
    try:
        float(str(s).strip().strip('"'))
        return True
    except ValueError:
        return False


def _basename(s: str) -> str:
    return re.split(r"[\\/]", str(s))[-1]


def _to_float(series: pd.Series) -> pd.Series:
    """Coerce to float; tolerates comma decimals and 'hh:mm:ss.fff' / 'mm:ss' time strings."""
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds().astype(float)
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    num = pd.to_numeric(series, errors="coerce").astype(float)
    todo = num.isna() & series.notna()
    if todo.any():
        text = series[todo].astype(str).str.strip()
        retry = pd.to_numeric(text.str.replace(",", ".", regex=False), errors="coerce")

        def _clock(v: str) -> float:
            parts = v.split(":")
            if not 2 <= len(parts) <= 3:
                return np.nan
            try:
                vals = [float(p.replace(",", ".")) for p in parts]
            except ValueError:
                return np.nan
            return float(sum(x * 60 ** i for i, x in enumerate(reversed(vals))))

        retry = retry.fillna(text.map(_clock))
        num.loc[todo] = retry.astype(float)
    return num


def _clean_str(series: pd.Series) -> pd.Series:
    """Object series of stripped strings with '', 'nan', 'None' -> NaN."""
    if series.isna().all():
        return pd.Series(np.nan, index=series.index, dtype=object)
    s = series.astype("string").str.strip()
    s = s.mask(s.isin(["", "nan", "NaN", "None", "<NA>", "null"]))
    return s.astype(object).where(s.notna(), np.nan)


_BINOMIAL_RE = re.compile(r"^[A-Z][a-z]+(?:[ _](?:[a-z]+(?:-[a-z]+)?|sp\.|spp\.|cf\.|x))+$")


def looks_like_scientific_name(s: str) -> bool:
    """Heuristic: 'Corvus corax', 'Anas sp.', 'Larus argentatus argenteus' -> True; 'Blue Tit' -> False."""
    return bool(_BINOMIAL_RE.match(str(s).strip()))


def split_label(label: Any) -> tuple[str | None, str | None]:
    """
    Derive (common_name, scientific_name) from a raw class label.

    Handles BirdNET "Scientific_Common" labels, old BirdNET Audacity "Scientific, Common" labels,
    bare scientific names ("Corvus corax"), underscored binomials ("Corvus_corax") and falls back
    to treating anything else (common names, species codes, "noise", ...) as the common name.
    """
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return None, None
    s = str(label).strip()
    if not s:
        return None, None

    if "_" in s:
        left, right = (p.strip() for p in s.split("_", 1))
        if left and right:
            lb, rb = looks_like_scientific_name(left), looks_like_scientific_name(right)
            if lb and not rb:
                return right, left
            if rb and not lb:
                return left, right
            if left.lower() == right.lower():          # BirdNET non-bird classes: "Dog_Dog"
                return right, left
            if lb and rb:                              # e.g. translated labels "Turdus merula_Merle noir"
                return right, left
        spaced = s.replace("_", " ")
        if looks_like_scientific_name(spaced):         # "Corvus_corax"
            return None, spaced

    if "," in s:
        left, right = (p.strip() for p in s.split(",", 1))
        lb, rb = looks_like_scientific_name(left), looks_like_scientific_name(right)
        if lb and not rb:
            return right, left
        if rb and not lb:
            return left, right

    if looks_like_scientific_name(s):
        return None, s
    return s, None


_LABEL_SCORE_RE = re.compile(r"^(.*?)\s*[;|]\s*(-?\d+(?:[.,]\d+)?(?:e-?\d+)?)\s*$")


def split_label_score(label: Any) -> tuple[Any, float]:
    """HawkEars style 'MAWR;0.88' -> ('MAWR', 0.88). Anything else -> (label, nan)."""
    if not isinstance(label, str):
        return label, np.nan
    m = _LABEL_SCORE_RE.match(label)
    if not m:
        return label, np.nan
    return m.group(1).strip(), float(m.group(2).replace(",", "."))


def infer_recording_stem(detections_path: Path) -> str:
    """'marsh_scores.txt' -> 'marsh'; 'rec1.BirdNET.selection.table.txt' -> 'rec1'."""
    stem = Path(detections_path).stem
    prev = None
    while prev != stem:
        prev = stem
        stem = _SUFFIX_RE.sub("", stem)
    return stem


# ---------------------------------------------------------------------------------------------
# Options and audio lookup
# ---------------------------------------------------------------------------------------------
LabelMap = Mapping[str, Any] | Callable[[str], Any]


@dataclass
class LoadOptions:
    """Options shared by every parser (see ``load_detections`` for the meaning of each)."""

    filename_mode: str = "as_is"                 # "as_is" | "basename" | "stem"
    min_confidence: float | None = None
    default_duration: float | None = None
    label_map: LabelMap | None = None
    column_map: Mapping[str, str] | None = None  # source column (raw or normalised) -> standard column
    keep_extra_columns: bool = False
    audio_dir: Path | None = None
    _audio_index: dict[str, str] | None = field(default=None, repr=False)
    _dir_cache: dict[Path, dict[str, str]] = field(default_factory=dict, repr=False)

    # -- audio file resolution ------------------------------------------------------------------
    def _dir_listing(self, folder: Path) -> dict[str, str]:
        """{lower-case audio filename: actual filename} for one folder (non-recursive)."""
        if folder not in self._dir_cache:
            listing: dict[str, str] = {}
            try:
                for p in folder.iterdir():
                    if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
                        listing[p.name.lower()] = p.name
            except OSError:
                pass
            self._dir_cache[folder] = listing
        return self._dir_cache[folder]

    def _audio_dir_index(self) -> dict[str, str]:
        """{lower-case stem or name: filename} for every audio file under ``audio_dir`` (recursive)."""
        if self._audio_index is None:
            index: dict[str, str] = {}
            if self.audio_dir is not None and Path(self.audio_dir).is_dir():
                for p in sorted(Path(self.audio_dir).rglob("*")):
                    if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
                        index.setdefault(p.name.lower(), p.name)
                        index.setdefault(p.stem.lower(), p.name)
            self._audio_index = index
        return self._audio_index

    def resolve_audio_name(self, stem: str, near: Path | None = None) -> str | None:
        """Find an audio file called ``<stem>.<audio ext>`` in audio_dir or next to ``near``."""
        key = stem.lower()
        if self.audio_dir is not None:
            hit = self._audio_dir_index().get(key)
            if hit:
                return hit
        if near is not None:
            for folder in (near.parent, near.parent.parent):
                listing = self._dir_listing(folder)
                for ext in AUDIO_EXTENSIONS:
                    hit = listing.get(key + ext)
                    if hit:
                        return hit
        return None

    def infer_filename(self, detections_path: Path) -> Any:
        """Best-effort recording name for detections files that carry no filename column."""
        stem = infer_recording_stem(detections_path)
        if stem.lower() in _COMBINED_STEMS:
            return np.nan                         # combined file across recordings: unknown
        if Path(stem).suffix.lower() in AUDIO_EXTENSIONS:
            return stem                           # "rec.wav.selections.txt" -> "rec.wav"
        return self.resolve_audio_name(stem, near=detections_path) or stem


# ---------------------------------------------------------------------------------------------
# Sniffing (cheap look at a file, used both for finding and for loading)
# ---------------------------------------------------------------------------------------------
@dataclass
class Sniff:
    path: Path
    ext: str
    encoding: str = "utf-8"
    text: str | None = None                       # head of the file (text formats only)
    lines: list[str] = field(default_factory=list)
    delimiter: str | None = None
    has_header: bool = False
    columns: list[str] = field(default_factory=list)
    table: pd.DataFrame | None = None             # pre-read table (parquet / json / xlsx)
    column_map: Mapping[str, str] | None = None   # user overrides, consulted by matchers

    @property
    def norm_columns(self) -> list[str]:
        return [_norm(c) for c in self.columns]

    @property
    def norm_set(self) -> set[str]:
        return set(self.norm_columns)

    @property
    def posix_lower(self) -> str:
        return self.path.as_posix().lower()

    def data_lines(self, n: int = 20) -> list[str]:
        rows = self.lines[1:] if self.has_header else self.lines
        return [ln for ln in rows if not ln.startswith("\\")][:n]


def _read_head(path: Path, min_bytes: int = 65536, max_bytes: int = 64 * 1024 * 1024) -> tuple[str, str, bool]:
    """Return (text, encoding, complete). Reads at least a few lines even for very wide tables."""
    chunks: list[bytes] = []
    total = 0
    complete = False
    with open(path, "rb") as f:
        while True:
            chunk = f.read(min_bytes)
            if not chunk:
                complete = True
                break
            chunks.append(chunk)
            total += len(chunk)
            if total >= min_bytes and b"".join(chunks).count(b"\n") >= 3:
                break
            if total >= max_bytes:
                break
    raw = b"".join(chunks)
    if raw.startswith(codecs.BOM_UTF16_LE) or raw.startswith(codecs.BOM_UTF16_BE):
        encoding = "utf-16"
    else:
        encoding = "latin-1"
        for cut in range(0, 4):                   # tolerate a multi-byte char cut at the boundary
            try:
                raw[: len(raw) - cut].decode("utf-8")
                encoding = "utf-8-sig"
                break
            except UnicodeDecodeError:
                continue
    text = raw.decode(encoding, errors="replace")
    if not complete:
        text = text[: text.rfind("\n") + 1]       # drop the trailing partial line
    return text, encoding, complete


def _detect_delimiter(lines: Sequence[str]) -> str | None:
    sample = [ln for ln in lines[:25] if not ln.startswith("\\")] or list(lines[:25])
    best, best_key = None, None
    for d in ("\t", ",", ";", "|"):
        counts = [ln.count(d) for ln in sample]
        if not counts or min(counts) == 0:
            continue
        consistent = len(set(counts)) == 1
        if d == "\t" and consistent:              # tabs inside data are rare: trust them
            return "\t"
        key = (consistent, counts[0])
        if best_key is None or key > best_key:
            best, best_key = d, key
    return best


def sniff_file(path: str | Path) -> Sniff:
    """Look at a file just enough to decide what it is (header, delimiter, columns)."""
    path = Path(path)
    ext = path.suffix.lower()
    sn = Sniff(path=path, ext=ext)
    if ext in _STRUCTURED_EXTENSIONS:
        sn.table = _read_structured(path, ext)
        sn.columns = [str(c).strip() for c in sn.table.columns]
        sn.has_header = True
        return sn
    text, encoding, _ = _read_head(path)
    sn.text, sn.encoding = text, encoding
    sn.lines = [ln for ln in text.splitlines() if ln.strip()]
    if not sn.lines:
        return sn
    sn.delimiter = _detect_delimiter(sn.lines)
    first = sn.lines[0]
    fields = first.split(sn.delimiter) if sn.delimiter else [first]
    looks_like_data = len(fields) >= 2 and _is_number(fields[0]) and _is_number(fields[1])
    sn.has_header = not looks_like_data
    if sn.has_header:
        sn.columns = [f.strip().strip('"').strip() for f in fields]
    return sn


# ---------------------------------------------------------------------------------------------
# Raw table readers
# ---------------------------------------------------------------------------------------------
def _read_json_table(path: Path, ext: str) -> pd.DataFrame:
    if ext in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    record_keys = ("detections", "predictions", "annotation", "annotations", "results",
                   "events", "selections", "labels", "data", "items", "records")
    if isinstance(data, list):
        if data and all(isinstance(d, dict) for d in data):
            nested = next((k for k in record_keys if all(isinstance(d.get(k), list) for d in data)), None)
            if nested is not None:            # [{"file": "a.wav", "detections": [...]}, ...]
                meta = [k for k in data[0] if not isinstance(data[0][k], (list, dict))]
                return pd.json_normalize(data, record_path=nested, meta=meta, errors="ignore")
        return pd.json_normalize(data)
    if isinstance(data, dict):
        for key in record_keys:
            if isinstance(data.get(key), list):
                df = pd.json_normalize(data[key])
                # carry file-level metadata (e.g. BatDetect2 "id": "<recording>.wav") onto the rows
                if not any(_norm(c) in COLUMN_SYNONYMS["filename"] for c in df.columns):
                    for k in ("filename", "file", "file_name", "filepath", "path", "recording",
                              "audio_file", "id"):
                        if k in data and not isinstance(data[k], (list, dict)):
                            df["filename"] = data[k]
                            break
                return df
        if data and all(isinstance(v, list) for v in data.values()):
            if all(all(isinstance(d, dict) for d in v) for v in data.values()):
                frames = []
                for key, rows in data.items():          # {"a.wav": [ {...}, ... ], ...}
                    f = pd.json_normalize(rows)
                    if not any(_norm(c) in COLUMN_SYNONYMS["filename"] for c in f.columns):
                        f["filename"] = key
                    frames.append(f)
                return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            return pd.DataFrame(data)                    # column-oriented {"col": [..], ...}
        return pd.json_normalize(data)
    raise ValueError(f"Unsupported JSON structure in {path}")


def _read_structured(path: Path, ext: str) -> pd.DataFrame:
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if ext == ".feather":
        return pd.read_feather(path)
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext in (".json", ".jsonl", ".ndjson"):
        return _read_json_table(path, ext)
    raise ValueError(f"Unsupported structured file type: {ext}")


def read_raw_table(sn: Sniff) -> pd.DataFrame:
    """Read the whole file as-is (header preserved) into a DataFrame."""
    if sn.table is not None:
        df = sn.table.copy()
    else:
        if sn.delimiter is None:
            raise ValueError(f"Could not detect a delimiter in {sn.path}")
        import csv as _csv

        df = pd.read_csv(
            sn.path,
            sep=sn.delimiter,
            header=0 if sn.has_header else None,
            encoding=sn.encoding,
            quoting=_csv.QUOTE_NONE if sn.delimiter == "\t" else _csv.QUOTE_MINIMAL,
            skip_blank_lines=True,
            low_memory=False,
            encoding_errors="replace",
            index_col=False,          # ragged rows must never promote column 1 to the index
            on_bad_lines="warn",
        )
    df.columns = [str(c).strip() for c in df.columns]
    return df


# ---------------------------------------------------------------------------------------------
# Column mapping (the generic heart of the loader)
# ---------------------------------------------------------------------------------------------
_ALL_SYNONYMS = {syn for syns in COLUMN_SYNONYMS.values() for syn in syns}


def pick_columns(columns: Iterable[str], column_map: Mapping[str, str] | None = None) -> dict[str, str]:
    """Map standard names -> raw column names for a header, using explicit overrides then synonyms."""
    columns = list(columns)
    by_norm: dict[str, str] = {}
    for c in columns:
        by_norm.setdefault(_norm(c), c)
    picked: dict[str, str] = {}
    used: set[str] = set()
    for src, std in (column_map or {}).items():
        raw = src if src in columns else by_norm.get(_norm(src))
        if raw is not None and std in COLUMN_SYNONYMS:
            picked[std] = raw
            used.add(raw)
    for std, syns in COLUMN_SYNONYMS.items():
        if std in picked:
            continue
        for syn in syns:
            raw = by_norm.get(syn)
            if raw is not None and raw not in used:
                picked[std] = raw
                used.add(raw)
                break
    # "onset"/"offset" are a start/end pair; "offset" alone (Kaleidoscope) is a start offset
    for off in ("offset", "offset_s"):
        raw = by_norm.get(off)
        if (raw is not None and raw not in used and "end_time" not in picked
                and "start_time" in picked and _norm(picked["start_time"]).startswith("onset")):
            picked["end_time"] = raw
            used.add(raw)
    return picked


def map_columns(raw: pd.DataFrame, column_map: Mapping[str, str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (standard-ish frame, leftover raw columns). Missing standard columns are left out."""
    picked = pick_columns(raw.columns, column_map)
    out = pd.DataFrame(index=raw.index)
    for std, src in picked.items():
        out[std] = raw[src]
    extras = raw[[c for c in raw.columns if c not in picked.values()]]
    return out, extras


# ---------------------------------------------------------------------------------------------
# Format registry
# ---------------------------------------------------------------------------------------------
class DetectionFormat:
    """
    Base class for a detections file format.

    Subclasses implement ``match`` (0 = cannot read this file, higher = better fit) and
    ``read`` (return a frame with any subset of ``STANDARD_COLUMNS`` plus, optionally,
    ``duration``; everything else is standardised afterwards).
    """

    name: str = "base"
    tool: str = "unknown"
    #: ((regex tested against the lower-case POSIX path, bonus), ...)
    filename_patterns: Sequence[tuple[str, int]] = ()

    def match(self, sn: Sniff) -> int:
        raise NotImplementedError

    def read(self, sn: Sniff, opts: LoadOptions) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    def filename_score(self, sn: Sniff) -> int:
        bonus = 0
        for pattern, value in self.filename_patterns:
            if re.search(pattern, sn.posix_lower):
                bonus = max(bonus, value)
        if "/rarities/" in sn.posix_lower:       # HawkEars rare-species side output
            bonus -= 5
        return bonus

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} tool={self.tool!r}>"


class TableFormat(DetectionFormat):
    """A header-based table. Declare which normalised columns identify it; mapping is generic."""

    required: frozenset[str] = frozenset()             # all of these normalised columns
    any_of: Sequence[frozenset[str]] = ()              # at least one column from each group
    base_score: int = 50
    column_map: Mapping[str, str] = {}                 # extra source -> standard overrides
    default_duration: float | None = None              # used when the table has no end/duration

    def match(self, sn: Sniff) -> int:
        if not sn.has_header or not sn.columns:
            return 0
        cols = sn.norm_set
        if not self.required <= cols:
            return 0
        for group in self.any_of:
            if not (group & cols):
                return 0
        return self.base_score + self.filename_score(sn)

    def prepare(self, raw: pd.DataFrame, sn: Sniff, opts: LoadOptions) -> pd.DataFrame:
        """Hook for format quirks on the raw table (before column mapping)."""
        return raw

    def read(self, sn: Sniff, opts: LoadOptions) -> tuple[pd.DataFrame, pd.DataFrame]:
        raw = self.prepare(read_raw_table(sn), sn, opts)
        column_map = dict(self.column_map)
        if opts.column_map:
            column_map.update(opts.column_map)
        df, extras = map_columns(raw, column_map)
        duration = opts.default_duration if opts.default_duration is not None else self.default_duration
        if "end_time" not in df and "duration" not in df and duration is not None and "start_time" in df:
            df["end_time"] = _to_float(df["start_time"]) + duration
        return df, extras


_FORMATS: list[DetectionFormat] = []


def register_format(cls: type[DetectionFormat]) -> type[DetectionFormat]:
    """Class decorator: register a DetectionFormat subclass (later registrations win ties)."""
    _FORMATS[:] = [f for f in _FORMATS if f.name != cls.name]
    _FORMATS.append(cls())
    return cls


def available_formats() -> list[DetectionFormat]:
    return list(_FORMATS)


def detect_format(sn: Sniff, format: str | None = None) -> tuple[DetectionFormat | None, int]:
    """Best (format, score) for a sniffed file; (None, 0) if nothing can read it."""
    if format is not None:
        for f in _FORMATS:
            if f.name == format:
                return f, max(f.match(sn), 1)     # explicit request: always readable
        raise ValueError(f"Unknown format {format!r}. Known: {[f.name for f in _FORMATS]}")
    best, best_key = None, (0, 0)
    for i, f in enumerate(_FORMATS):
        try:
            score = f.match(sn)
        except Exception:                          # a broken matcher must not break detection
            score = 0
        if score > 0 and (score, i) > best_key:
            best, best_key = f, (score, i)
    return best, best_key[0]


# ---------------------------------------------------------------------------------------------
# Built-in formats
# ---------------------------------------------------------------------------------------------
@register_format
class GenericTable(TableFormat):
    """Any table with a recognisable start-time column plus something to label/score it with."""

    name = "generic"
    tool = "unknown"
    base_score = 30

    def match(self, sn: Sniff) -> int:
        if not sn.has_header or not sn.columns:
            return 0
        picked = pick_columns(sn.columns, sn.column_map)
        if "start_time" not in picked:
            return 0
        if not ({"end_time", "duration", "label", "common_name", "scientific_name", "confidence"}
                & picked.keys()):
            return 0
        return self.base_score + self.filename_score(sn)


@register_format
class WideTable(TableFormat):
    """
    OpenSoundscape-style wide table: identifier columns (file, start_time, end_time) followed by
    one numeric score column per class. Melted into long format; use ``min_confidence`` to keep
    the result small.
    """

    name = "wide"
    tool = "OpenSoundscape / wide scores table"
    base_score = 35

    def _class_columns(self, columns: Sequence[str]) -> list[str]:
        picked = pick_columns(columns)
        used = set(picked.values())
        return [c for c in columns if c not in used and _norm(c) not in _NON_CLASS_COLUMNS
                and _norm(c) not in _ALL_SYNONYMS]

    def match(self, sn: Sniff) -> int:
        if not sn.has_header or not sn.columns:
            return 0
        picked = pick_columns(sn.columns, sn.column_map)
        if "start_time" not in picked:
            return 0
        if {"label", "confidence", "common_name", "scientific_name"} & picked.keys():
            return 0
        classes = self._class_columns(sn.columns)
        if not classes:
            return 0
        if sn.table is not None:
            numeric = [c for c in classes if pd.api.types.is_numeric_dtype(sn.table[c])]
            if not numeric:
                return 0
        else:
            rows = sn.data_lines(5)
            if rows:
                idx = [sn.columns.index(c) for c in classes]
                for row in rows:
                    fields = row.split(sn.delimiter)
                    vals = [fields[i] for i in idx if i < len(fields)]
                    if not vals or not all(_is_number(v) or v.strip() == "" for v in vals):
                        return 0
        return self.base_score + self.filename_score(sn)

    def read(self, sn: Sniff, opts: LoadOptions) -> tuple[pd.DataFrame, pd.DataFrame]:
        raw = read_raw_table(sn)
        picked = pick_columns(raw.columns, opts.column_map)
        classes = [c for c in self._class_columns(list(raw.columns))
                   if pd.api.types.is_numeric_dtype(pd.to_numeric(raw[c], errors="coerce"))]
        ids = pd.DataFrame({std: raw[src] for std, src in picked.items()})
        values = raw[classes].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        mask = ~np.isnan(values)
        if opts.min_confidence is not None:
            mask &= values >= opts.min_confidence
        rows, cols = np.nonzero(mask)
        out = ids.iloc[rows].reset_index(drop=True)
        out["label"] = np.asarray(classes, dtype=object)[cols]
        out["confidence"] = values[rows, cols]
        return out, pd.DataFrame(index=out.index)


@register_format
class AudacityLabels(DetectionFormat):
    """
    Headerless Audacity label track: ``start<TAB>end<TAB>label[<TAB>confidence]``.
    Written by BirdNET (label = "Scientific_Common", 4th column = confidence) and HawkEars
    (label = "NAME;score"). Frequency lines starting with a backslash are ignored.
    """

    name = "audacity"
    tool = "Audacity labels"
    filename_patterns = (
        (r"birdnet_audacitylabels\.txt$", 25),
        (r"\.birdnet\.results\.txt$", 20),
        (r"_scores\.txt$", 20),               # HawkEars 2.x
        (r"_hawkears\.txt$", 20),             # HawkEars 1.x
        (r"\.(txt|lbl)$", 5),
    )

    def match(self, sn: Sniff) -> int:
        if sn.text is None or sn.has_header:
            return 0
        if sn.ext not in {".txt", ".tsv", ".csv", ".lbl", ".tab"}:
            return 0
        bonus = self.filename_score(sn)
        rows = sn.data_lines(30)
        if not rows:                          # empty file (e.g. HawkEars recording with 0 detections)
            return 40 + bonus if bonus >= 20 else 0
        for row in rows:
            fields = row.split("\t") if "\t" in row else row.split(",")
            if len(fields) < 3 or not (_is_number(fields[0]) and _is_number(fields[1])):
                return 0
        return 40 + bonus

    def read(self, sn: Sniff, opts: LoadOptions) -> pd.DataFrame:
        import csv as _csv

        sep = sn.delimiter if sn.delimiter in ("\t", ",", ";") else "\t"
        if not sn.lines:                        # empty label file: recording had no detections
            df = pd.DataFrame(columns=["start_time", "end_time", "label", "confidence"])
        else:
            raw = pd.read_csv(
                sn.path, sep=sep, header=None, names=list(range(8)), index_col=False,
                dtype=str, encoding=sn.encoding, encoding_errors="replace",
                quoting=_csv.QUOTE_NONE if sep == "\t" else _csv.QUOTE_MINIMAL,
                skip_blank_lines=True, on_bad_lines="warn",
            )
            raw = raw.dropna(axis=1, how="all")
            df = pd.DataFrame({
                "start_time": raw.iloc[:, 0] if raw.shape[1] > 0 else np.nan,
                "end_time": raw.iloc[:, 1] if raw.shape[1] > 1 else np.nan,
                "label": raw.iloc[:, 2] if raw.shape[1] > 2 else np.nan,
                "confidence": raw.iloc[:, 3] if raw.shape[1] > 3 else np.nan,
            })
            # drop Audacity frequency-range lines ("\\<TAB>low<TAB>high") and anything non-numeric
            df = df[pd.to_numeric(df["start_time"], errors="coerce").notna()].reset_index(drop=True)
        df["filename"] = opts.infer_filename(sn.path)
        return df


@register_format
class RavenSelectionTable(TableFormat):
    """
    Raven Pro selection table (also what BirdNET / HawkEars write with ``--rtype table/raven``).
    Times are made relative to the recording via ``File Offset (s)`` when present, and
    duplicate rows per View (Waveform / Spectrogram) are collapsed.
    """

    name = "raven"
    tool = "Raven Pro"
    required = frozenset({"begin_time", "end_time"})
    any_of = (frozenset({"selection", "view", "channel", "low_freq", "high_freq", "file_offset",
                         "begin_file", "begin_path", "delta_time"}),)
    base_score = 60
    filename_patterns = (
        (r"\.selections\.txt$", 20),
        (r"\.selection\.table\.txt$", 15),
    )

    _RAVEN_STANDARD = frozenset({
        "selection", "view", "channel", "begin_time", "end_time", "low_freq", "high_freq",
        "delta_time", "delta_freq", "begin_file", "end_file", "begin_path", "end_path",
        "file_offset", "begin_date", "begin_clock_time", "end_clock_time", "begin_hour",
        "delta_time_s", "notes", "comments", "peak_freq", "center_freq", "max_freq", "min_freq",
        "avg_power_density", "max_power_density", "peak_power_density", "energy", "agg_entropy",
        "avg_entropy", "bw_90", "center_time", "dur_90", "freq_5", "freq_95", "time_5", "time_95",
        "inband_power", "leq", "sel", "snr_nist_quick", "peak_time", "max_time", "min_time",
        "dur_50", "bw_50", "freq_25", "freq_75", "time_25", "time_75", "iqr_bw", "iqr_dur",
        "peak_amp", "rms_amp", "max_amp", "min_amp", "filtered_rms_amp", "occupancy",
        "length_frames", "sample_rate", "begin_sample", "end_sample", "length_samples",
    })

    def prepare(self, raw: pd.DataFrame, sn: Sniff, opts: LoadOptions) -> pd.DataFrame:
        by_norm = {_norm(c): c for c in raw.columns}
        raw = raw.copy()
        # annotators name their label column freely: if nothing recognisable exists, take the
        # first non-measurement text column (Raven measurement columns are all numeric)
        picked = pick_columns(raw.columns, opts.column_map)
        if not ({"label", "common_name", "scientific_name"} & picked.keys()):
            for col in raw.columns:
                n = _norm(col)
                if col in picked.values() or n in self._RAVEN_STANDARD or n in _ALL_SYNONYMS:
                    continue
                if pd.api.types.is_numeric_dtype(raw[col]) or raw[col].notna().mean() < 0.5:
                    continue
                raw = raw.rename(columns={col: "label"})
                break
        # one selection can be listed once per view: keep a single row (prefer the spectrogram)
        if "selection" in by_norm and "view" in by_norm:
            sel = by_norm["selection"]
            if raw[sel].duplicated().any():
                view = raw[by_norm["view"]].astype(str).str.lower()
                subset = [sel] + [by_norm[c] for c in ("begin_file", "begin_path") if c in by_norm]
                raw = (raw.assign(_pref=(~view.str.contains("spectrogram")).astype(int))
                          .sort_values("_pref", kind="stable")
                          .drop_duplicates(subset=subset, keep="first")
                          .drop(columns="_pref")
                          .sort_index())
        # multi-file tables: Begin Time counts from the start of the *sequence*
        if "file_offset" in by_norm:
            begin, end, off = (by_norm[c] for c in ("begin_time", "end_time", "file_offset"))
            b, e, o = _to_float(raw[begin]), _to_float(raw[end]), _to_float(raw[off])
            ok = o.notna()
            raw.loc[ok, begin] = o[ok]
            raw.loc[ok, end] = (o + (e - b))[ok]
            raw = raw.drop(columns=off)
        return raw


@register_format
class BirdNETRavenTable(RavenSelectionTable):
    name = "birdnet_raven"
    tool = "BirdNET-Analyzer"
    required = frozenset({"begin_time", "end_time", "common_name", "confidence", "species_code"})
    base_score = 75
    filename_patterns = (
        (r"birdnet_selectiontable\.txt$", 30),
        (r"\.birdnet\.selection\.table\.txt$", 20),
        (r"\.selections\.txt$", 5),
    )


@register_format
class HawkEarsRavenTable(RavenSelectionTable):
    name = "hawkears_raven"
    tool = "HawkEars"
    required = frozenset({"begin_time", "end_time", "common_name", "confidence", "ebird_code"})
    base_score = 76
    filename_patterns = ((r"\.hawkears\.selection\.table\.txt$", 20),)


@register_format
class BirdNETCSV(TableFormat):
    """BirdNET_CombinedTable.csv / <rec>.BirdNET.results.csv (and the .parquet twins)."""

    name = "birdnet_csv"
    tool = "BirdNET-Analyzer"
    required = frozenset({"start", "end", "scientific_name", "common_name", "confidence"})
    base_score = 80
    filename_patterns = (
        (r"birdnet_combinedtable\.(csv|parquet)$", 30),
        (r"\.birdnet\.results\.(csv|parquet)$", 20),
    )


@register_format
class BirdNETRTable(TableFormat):
    """Legacy BirdNET ``--rtype r`` table: filepath,start,end,scientific_name,common_name,confidence,..."""

    name = "birdnet_rtable"
    tool = "BirdNET-Analyzer"
    required = frozenset({"filepath", "start", "end", "scientific_name", "common_name", "confidence"})
    base_score = 81
    filename_patterns = ((r"birdnet_rtable\.csv$", 30),)


@register_format
class BirdNETLibTable(TableFormat):
    """birdnetlib / birdnet python package style rows: common_name, scientific_name, start_time, end_time, confidence."""

    name = "birdnetlib"
    tool = "birdnetlib"
    required = frozenset({"start_time", "end_time", "scientific_name", "common_name", "confidence"})
    base_score = 70


@register_format
class KaleidoscopeCSV(TableFormat):
    """Wildlife Acoustics Kaleidoscope (id.csv / cluster.csv) and BirdNET's Kaleidoscope export."""

    name = "kaleidoscope"
    tool = "Kaleidoscope"
    required = frozenset({"in_file", "offset"})
    any_of = (frozenset({"duration", "top1match", "top1dist", "auto_id", "manual_id",
                         "common_name", "confidence"}),)
    base_score = 60
    filename_patterns = (
        (r"birdnet_kaleidoscope\.csv$", 30),
        (r"\.birdnet\.results\.kaleidoscope\.csv$", 20),
        (r"(^|/)(id|cluster|meta)\.csv$", 10),
    )

    def prepare(self, raw: pd.DataFrame, sn: Sniff, opts: LoadOptions) -> pd.DataFrame:
        by_norm = {_norm(c): c for c in raw.columns}
        raw = raw.copy()
        parts = [by_norm[c] for c in ("indir", "folder", "in_file") if c in by_norm]
        if len(parts) > 1:
            def _join(row):
                pieces = [str(v).strip() for v in row if pd.notna(v) and str(v).strip() not in ("", ".")]
                if not pieces:
                    return np.nan
                sep = "\\" if ("\\" in pieces[0] and "/" not in pieces[0]) else "/"
                return sep.join(p.rstrip("/\\") for p in pieces)
            raw["__kaleidoscope_path__"] = raw[parts].apply(_join, axis=1)
            raw = raw.drop(columns=[by_norm["in_file"]])
        return raw

    column_map = {"__kaleidoscope_path__": "filename"}


@register_format
class HawkEarsCSV(TableFormat):
    """HawkEars 2.x scores.csv / rarities.csv: recording,name,start_time,end_time,score."""

    name = "hawkears_csv"
    tool = "HawkEars"
    required = frozenset({"recording", "name", "start_time", "end_time", "score"})
    base_score = 80
    filename_patterns = ((r"(^|/)scores\.csv$", 30), (r"(^|/)rarities\.csv$", 10))


@register_format
class BirdCODESelectionTable(TableFormat):
    """Earth Species Project sound-event-detection (BirdCODE): Begin Time (s), End Time (s), Species, Score."""

    name = "birdcode"
    tool = "BirdCODE (earthspecies/sound-event-detection)"
    required = frozenset({"begin_time", "end_time", "species", "score"})
    base_score = 70
    filename_patterns = ((r"birdcode_predictions/", 20),)


@register_format
class PerchInferenceCSV(TableFormat):
    """Perch / chirp ``write_inference_csv``: filename, timestamp_s, label, logit (5 s windows)."""

    name = "perch_chirp_csv"
    tool = "Perch (chirp inference)"
    required = frozenset({"filename", "timestamp_s", "label", "logit"})
    base_score = 70
    default_duration = 5.0


@register_format
class PerchHopliteCSV(TableFormat):
    """perch-hoplite agile classifier csv: idx,project,filename,window_start,window_end,label,logits."""

    name = "perch_hoplite_csv"
    tool = "Perch (hoplite agile classifier)"
    required = frozenset({"filename", "window_start", "window_end", "label", "logits"})
    base_score = 70


# ---------------------------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------------------------
def _apply_label_map(df: pd.DataFrame, label_map: LabelMap) -> pd.DataFrame:
    """Resolve raw labels through a user mapping; only fills names, never overwrites label."""
    def _lookup(label: str) -> tuple[Any, Any]:
        res = label_map(label) if callable(label_map) else label_map.get(label)
        if res is None:
            return None, None
        if isinstance(res, str):
            return (None, res) if looks_like_scientific_name(res) else (res, None)
        if isinstance(res, Mapping):
            return res.get("common_name"), res.get("scientific_name")
        return tuple(res)[:2]

    labels = df["label"]
    has = labels.notna()
    resolved = {lbl: _lookup(lbl) for lbl in labels[has].unique()}
    common = labels[has].map(lambda l: resolved[l][0])
    sci = labels[has].map(lambda l: resolved[l][1])
    # a mapping is authoritative: it overrides names that were merely guessed from the label
    df.loc[common.index[common.notna()], "common_name"] = common[common.notna()]
    df.loc[sci.index[sci.notna()], "scientific_name"] = sci[sci.notna()]
    return df


def _standardise(df: pd.DataFrame, extras: pd.DataFrame | None, sn: Sniff,
                 fmt: DetectionFormat, opts: LoadOptions) -> pd.DataFrame:
    df = df.copy()
    df = df.reset_index(drop=True)
    if extras is not None:
        extras = extras.reset_index(drop=True)

    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # --- text columns --------------------------------------------------------------------------
    for col in ("filename", "common_name", "scientific_name", "label"):
        df[col] = _clean_str(df[col])

    # --- confidence embedded in the label ("MAWR;0.88", HawkEars Audacity labels) -------------
    df["confidence"] = _to_float(df["confidence"])
    if df["label"].notna().any():
        lab = df["label"].astype("string")
        if lab.str.contains(r"[;|]", regex=True).fillna(False).any():
            parts = lab.str.extract(_LABEL_SCORE_RE.pattern)      # vectorised: (label, score)
            has_embedded = parts[1].notna()
            if has_embedded.any():
                df.loc[has_embedded, "label"] = parts.loc[has_embedded, 0].str.strip().astype(object)
                embedded = pd.to_numeric(parts[1].str.replace(",", ".", regex=False), errors="coerce")
                fill = has_embedded & df["confidence"].isna()
                df.loc[fill, "confidence"] = embedded[fill].astype(float)

    # --- derive names from the raw label -----------------------------------------------------
    need = df["label"].notna() & (df["common_name"].isna() | df["scientific_name"].isna())
    if need.any():
        derived = {lbl: split_label(lbl) for lbl in df.loc[need, "label"].unique()}
        common = df.loc[need, "label"].map(lambda l: derived[l][0])
        sci = df.loc[need, "label"].map(lambda l: derived[l][1])
        fill_c = need & df["common_name"].isna()
        fill_s = need & df["scientific_name"].isna()
        df.loc[fill_c, "common_name"] = common[fill_c[need]]
        df.loc[fill_s, "scientific_name"] = sci[fill_s[need]]
    if opts.label_map is not None and df["label"].notna().any():
        df = _apply_label_map(df, opts.label_map)
    for col in ("common_name", "scientific_name"):
        df[col] = _clean_str(df[col])

    # --- times ---------------------------------------------------------------------------------
    df["start_time"] = _to_float(df["start_time"])
    df["end_time"] = _to_float(df["end_time"])
    if "duration" in df.columns:
        dur = _to_float(df["duration"])
        fill = df["end_time"].isna() & dur.notna()
        df.loc[fill, "end_time"] = df.loc[fill, "start_time"] + dur[fill]
        df = df.drop(columns="duration")
    if opts.default_duration is not None:
        fill = df["end_time"].isna() & df["start_time"].notna()
        df.loc[fill, "end_time"] = df.loc[fill, "start_time"] + opts.default_duration

    # --- filename ------------------------------------------------------------------------------
    if df["filename"].isna().all():
        df["filename"] = opts.infer_filename(sn.path)
    elif opts.audio_dir is not None:
        # names without an audio extension (HawkEars writes stems): resolve against audio_dir
        def _resolve(name: Any) -> Any:
            if not isinstance(name, str) or Path(name).suffix.lower() in AUDIO_EXTENSIONS:
                return name
            return opts.resolve_audio_name(_basename(name)) or name
        df["filename"] = df["filename"].map(_resolve)
    if opts.filename_mode == "basename":
        df["filename"] = df["filename"].map(lambda v: _basename(v) if isinstance(v, str) else v)
    elif opts.filename_mode == "stem":
        df["filename"] = df["filename"].map(
            lambda v: Path(_basename(v)).stem if isinstance(v, str) else v)
    elif opts.filename_mode != "as_is":
        raise ValueError("filename_mode must be 'as_is', 'basename' or 'stem'")

    # --- confidence filter ---------------------------------------------------------------------
    keep = pd.Series(True, index=df.index)
    if opts.min_confidence is not None:
        keep &= ~(df["confidence"] < opts.min_confidence)     # NaN confidences are kept

    # --- assemble ------------------------------------------------------------------------------
    out = df[STANDARD_COLUMNS]
    if opts.keep_extra_columns and extras is not None and len(extras.columns):
        extras = extras.copy()
        extras.columns = [c if c not in STANDARD_COLUMNS else f"raw_{c}" for c in extras.columns]
        out = pd.concat([out, extras], axis=1)
    out = out.loc[keep].reset_index(drop=True)
    out.attrs.update({"source_format": fmt.name, "tool": fmt.tool, "source_files": [str(sn.path)]})
    return out


def _empty_frame() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype=float if c in ("start_time", "end_time", "confidence")
                                    else object) for c in STANDARD_COLUMNS})
    return df


def load_label_map(path: str | Path) -> dict[str, tuple[str | None, str | None]]:
    """
    Build a ``label_map`` (raw label -> (common_name, scientific_name)) from a class list file.

    Understands:
    * HawkEars ``classes.csv`` (Name,Code,AltName,AltCode): every column becomes a key
    * BirdNET label files (one ``Scientific_Common`` per line)
    * any csv/tsv with recognisable common/scientific/code/label columns
    """
    path = Path(path)
    out: dict[str, tuple[str | None, str | None]] = {}
    sn = sniff_file(path)
    if not sn.has_header:                                   # BirdNET labels txt
        with open(path, encoding=sn.encoding, errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    common, sci = split_label(line)
                    out[line] = (common, sci)
                    if sci:
                        out.setdefault(sci, (common, sci))
                    if common:
                        out.setdefault(common, (common, sci))
        return out
    raw = read_raw_table(sn)
    by_norm = {_norm(c): c for c in raw.columns}
    common_col = next((by_norm[c] for c in ("name", "common_name", "commonname", "english_name") if c in by_norm), None)
    sci_col = next((by_norm[c] for c in ("altname", "scientific_name", "scientificname", "sci_name") if c in by_norm), None)
    code_cols = [by_norm[c] for c in ("code", "altcode", "species_code", "ebird_code", "banding_code",
                                      "label", "class", "abbreviation") if c in by_norm]
    for _, row in raw.iterrows():
        common = str(row[common_col]).strip() if common_col and pd.notna(row[common_col]) else None
        sci = str(row[sci_col]).strip() if sci_col and pd.notna(row[sci_col]) else None
        for key_col in [common_col, sci_col, *code_cols]:
            if key_col is not None and pd.notna(row[key_col]):
                out.setdefault(str(row[key_col]).strip(), (common, sci))
    return out


# ---------------------------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------------------------
def describe_file(path: str | Path) -> dict[str, Any]:
    """Debug helper: which format would be used for this file, and why."""
    sn = sniff_file(path)
    fmt, score = detect_format(sn)
    return {
        "path": str(sn.path),
        "format": fmt.name if fmt else None,
        "tool": fmt.tool if fmt else None,
        "score": score,
        "delimiter": sn.delimiter,
        "has_header": sn.has_header,
        "columns": sn.columns,
        "mapped_columns": pick_columns(sn.columns) if sn.has_header else {},
        "all_scores": {f.name: f.match(sn) for f in _FORMATS if f.match(sn) > 0},
    }


def find_detections_files(
    folder: str | Path,
    *,
    recursive: bool = True,
    format: str | None = None,
    pattern: str | None = None,
    all_matches: bool = False,
) -> list[Path]:
    """
    Find detections files in ``folder``.

    Every candidate file (by extension) is sniffed and scored by the registered formats.
    By default only the files of the best-scoring (format, score) group are returned, so a
    folder holding e.g. ``BirdNET_CombinedTable.csv`` *and* per-recording Audacity labels yields
    just the combined table, while a folder of per-recording BirdCODE tables yields all of them.

    Parameters
    ----------
    recursive    also look in sub-folders (hidden folders are skipped)
    format       restrict to one registered format name (e.g. "raven")
    pattern      glob pattern to restrict candidates (e.g. "*.csv")
    all_matches  return every readable file, best first, instead of only the best group
    """
    folder = Path(folder)
    if folder.is_file():
        return [folder]
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a directory")
    glob = pattern or "*"
    candidates = folder.rglob(glob) if recursive else folder.glob(glob)
    scored: list[tuple[int, str, Path]] = []
    for path in sorted(candidates):
        if not path.is_file() or path.suffix.lower() not in TABLE_EXTENSIONS:
            continue
        if path.name.lower() in IGNORED_FILENAMES or path.name.startswith("."):
            continue
        if any(part.startswith(".") for part in path.relative_to(folder).parts[:-1]):
            continue
        try:
            sn = sniff_file(path)
            fmt, score = detect_format(sn, format=format)
        except Exception as exc:                   # unreadable / not a table: skip
            warnings.warn(f"Skipping {path}: {exc}", stacklevel=2)
            continue
        if fmt is None or score <= 0:
            continue
        scored.append((score, fmt.name, path))
    if not scored:
        return []
    scored.sort(key=lambda t: (-t[0], t[2].as_posix()))
    if all_matches:
        return [p for _, _, p in scored]
    best_score, best_name, _ = scored[0]
    return [p for s, n, p in scored if s == best_score and n == best_name]


def find_detections_file(folder: str | Path, **kwargs: Any) -> Path:
    """
    Return the single most likely detections file in ``folder`` (see ``find_detections_files``).

    Raises ``FileNotFoundError`` when nothing readable is found. When a folder holds one table
    per recording, this returns the first one; use ``find_detections_files`` or pass the folder
    itself to ``load_detections`` to get all of them.
    """
    files = find_detections_files(folder, **kwargs)
    if not files:
        raise FileNotFoundError(
            f"No detections file recognised in {folder}. Known formats: "
            f"{[f.name for f in _FORMATS]}. Use describe_file(path) on a candidate to see why."
        )
    if len(files) > 1:
        warnings.warn(
            f"{len(files)} detections files of the same format found in {folder}; returning "
            f"{files[0].name}. Use find_detections_files() or load_detections(folder) for all of them.",
            stacklevel=2,
        )
    return files[0]


def _load_one(path: Path, opts: LoadOptions, format: str | None) -> pd.DataFrame:
    sn = sniff_file(path)
    sn.column_map = opts.column_map
    fmt, score = detect_format(sn, format=format)
    if fmt is None:
        raise ValueError(
            f"Could not recognise {path} as a detections file "
            f"(header={sn.has_header}, columns={sn.columns[:12]}). "
            f"Pass format=... or column_map=... , or register a DetectionFormat for it."
        )
    result = fmt.read(sn, opts)
    df, extras = result if isinstance(result, tuple) else (result, None)
    if len(df) == 0:
        out = _empty_frame()
        out.attrs.update({"source_format": fmt.name, "tool": fmt.tool, "source_files": [str(path)]})
        return out
    return _standardise(df, extras, sn, fmt, opts)


def load_detections(
    source: str | Path | Sequence[str | Path],
    *,
    format: str | None = None,
    filename_mode: str = "as_is",
    min_confidence: float | None = None,
    default_duration: float | None = None,
    label_map: LabelMap | None = None,
    column_map: Mapping[str, str] | None = None,
    keep_extra_columns: bool = False,
    audio_dir: str | Path | None = None,
    recursive: bool = True,
) -> pd.DataFrame:
    """
    Load a detections file (or several, or a folder of them) into the standard DataFrame.

    Parameters
    ----------
    source            path to a detections file, a list of such paths, or a folder
                      (folders are searched with ``find_detections_files``; all files of the
                      best group are loaded and concatenated)
    format            force a registered format name instead of auto-detecting
    filename_mode     "as_is" (whatever the tool wrote), "basename" (strip directories) or
                      "stem" (also strip the extension) - useful to join outputs of different tools
    min_confidence    drop rows with confidence below this (rows without a confidence are kept)
    default_duration  end_time = start_time + default_duration when a table has no end/duration
                      (Perch chirp tables already default to 5 s)
    label_map         dict or callable resolving raw labels (e.g. HawkEars codes) to names:
                      value may be a scientific/common name string, a (common, scientific) tuple
                      or a {"common_name":..., "scientific_name":...} mapping
    column_map        extra {source column: standard column} overrides for odd headers
    keep_extra_columns  append the tool's remaining original columns after the standard ones
    audio_dir         folder with the original recordings; used to turn inferred stems
                      (e.g. "rec1" from "rec1_scores.txt") into real filenames ("rec1.wav")
    recursive         when ``source`` is a folder, search sub-folders too

    Returns
    -------
    DataFrame with columns ``STANDARD_COLUMNS`` (``filename, start_time, end_time, common_name,
    scientific_name, confidence, label``). ``df.attrs`` records ``source_format``, ``tool`` and
    ``source_files``.
    """
    opts = LoadOptions(
        filename_mode=filename_mode,
        min_confidence=min_confidence,
        default_duration=default_duration,
        label_map=label_map,
        column_map=column_map,
        keep_extra_columns=keep_extra_columns,
        audio_dir=Path(audio_dir) if audio_dir is not None else None,
    )
    if isinstance(source, (list, tuple, set)):
        paths = [Path(p) for p in source]
    else:
        src = Path(source)
        if src.is_dir():
            paths = find_detections_files(src, recursive=recursive, format=format)
            if not paths:
                raise FileNotFoundError(f"No detections file recognised in {src}")
        elif src.is_file():
            paths = [src]
        else:
            raise FileNotFoundError(str(src))

    frames = [_load_one(p, opts, format) for p in paths]
    if len(frames) == 1:
        return frames[0]
    non_empty = [f for f in frames if len(f)] or frames[:1]
    df = pd.concat(non_empty, ignore_index=True, sort=False)
    formats = sorted({f.attrs.get("source_format", "?") for f in frames})
    tools = sorted({f.attrs.get("tool", "?") for f in frames})
    df.attrs.update({
        "source_format": formats[0] if len(formats) == 1 else formats,
        "tool": tools[0] if len(tools) == 1 else tools,
        "source_files": [str(p) for p in paths],
    })
    return df
