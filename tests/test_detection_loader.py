"""Run with:  python -m pytest tests/ -q   (fixtures are created by tests/make_fixtures.py)"""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import detection_loader as dl  # noqa: E402

FIX = Path(__file__).parent / "fixtures"
if not FIX.exists():
    subprocess.run([sys.executable, str(Path(__file__).parent / "make_fixtures.py")], check=True)


def _check_standard(df: pd.DataFrame):
    assert list(df.columns[:7]) == dl.STANDARD_COLUMNS
    assert df["start_time"].dtype == float and df["end_time"].dtype == float
    assert df["confidence"].dtype == float
    assert (df["end_time"].fillna(np.inf) >= df["start_time"]).all()


@pytest.mark.parametrize("folder, expected_format, n_rows", [
    ("birdnet_combined", "birdnet_csv", 3),
    ("birdnet_parquet", "birdnet_csv", 2),
    ("birdnet_split", "birdnet_csv", 2),
    ("birdnet_split_raven", "birdnet_raven", 2),
    ("birdnet_raven_combined", "birdnet_raven", 3),
    ("birdnet_split_audacity", "audacity", 3),
    ("birdnet_kaleidoscope", "kaleidoscope", 2),
    ("birdnet_rtable", "birdnet_rtable", 1),
    ("hawkears_csv", "hawkears_csv", 3),
    ("hawkears_audacity", "audacity", 3),
    ("hawkears_audacity_names", "audacity", 1),
    ("hawkears_raven", "hawkears_raven", 1),
    ("birdcode", "birdcode", 3),
    ("perch_chirp", "perch_chirp_csv", 3),
    ("perch_hoplite", "perch_hoplite_csv", 2),
    ("raven", "raven", 2),
    ("raven_multi", "raven", 2),
    ("kaleidoscope", "kaleidoscope", 1),
    ("opensoundscape", "wide", 6),
    ("generic_json", "generic", 1),
    ("generic_semicolon", "generic", 1),
    ("generic_clock", "generic", 1),
])
def test_folder_roundtrip(folder, expected_format, n_rows):
    path = dl.find_detections_file(FIX / folder)
    df = dl.load_detections(FIX / folder)
    assert df.attrs["source_format"] == expected_format, dl.describe_file(path)
    assert len(df) == n_rows
    _check_standard(df)


def test_find_prefers_combined_table_and_ignores_params_file():
    files = dl.find_detections_files(FIX / "birdnet_combined")
    assert [f.name for f in files] == ["BirdNET_CombinedTable.csv"]
    everything = dl.find_detections_files(FIX / "birdnet_combined", all_matches=True)
    assert {f.name for f in everything} == {"BirdNET_CombinedTable.csv", "BirdNET_AudacityLabels.txt"}


def test_find_returns_all_per_recording_files():
    files = dl.find_detections_files(FIX / "birdcode")
    assert [f.name for f in files] == ["20230730.txt", "20260623.txt"]


def test_find_nothing_in_noise():
    assert dl.find_detections_files(FIX / "noise") == []
    with pytest.raises(FileNotFoundError):
        dl.find_detections_file(FIX / "noise")


def test_birdnet_csv_values():
    df = dl.load_detections(FIX / "birdnet_combined")
    r = df.iloc[0]
    assert r.filename == "/data/site1/rec1.wav" and r.start_time == 0.0 and r.end_time == 3.0
    assert r.common_name == "Eurasian Blackbird" and r.scientific_name == "Turdus merula"
    assert r.confidence == pytest.approx(0.8123)


def test_birdnet_audacity_label_split():
    df = dl.load_detections(FIX / "birdnet_combined/BirdNET_AudacityLabels.txt")
    assert df.common_name.tolist() == ["Eurasian Blackbird", "European Robin"]
    assert df.scientific_name.tolist() == ["Turdus merula", "Erithacus rubecula"]
    assert df.filename.isna().all()          # combined file: recording unknown


def test_birdnet_combined_raven_uses_file_offset():
    df = dl.load_detections(FIX / "birdnet_raven_combined")
    assert df.start_time.tolist() == [0.0, 3.0, 0.0]
    assert df.end_time.tolist() == [3.0, 6.0, 3.0]


def test_hawkears_audacity_embedded_score_and_audio_resolution():
    df = dl.load_detections(FIX / "hawkears_audacity")
    assert df.filename.tolist() == ["marsh.wav", "marsh.wav", "night.mp3"]
    assert df.confidence.tolist() == [0.88, 0.91, 0.75]
    assert df.label.tolist() == ["MAWR", "COYE", "CONI"]
    # rarities sub-folder is scored lower and not mixed in; empty label file loads as empty frame
    empty = dl.load_detections(FIX / "hawkears_audacity/quiet_scores.txt")
    assert len(empty) == 0 and list(empty.columns) == dl.STANDARD_COLUMNS


def test_hawkears_codes_resolved_with_label_map(tmp_path):
    classes = tmp_path / "classes.csv"
    classes.write_text("Name,Code,AltName,AltCode\nMarsh Wren,MAWR,Cistothorus palustris,marwre\n"
                       "Common Yellowthroat,COYE,Geothlypis trichas,comyel\n")
    lm = dl.load_label_map(classes)
    df = dl.load_detections(FIX / "hawkears_csv/scores.csv", label_map=lm, audio_dir=FIX / "hawkears_audacity")
    assert df.iloc[0].common_name == "Common Yellowthroat"
    assert df.iloc[0].scientific_name == "Geothlypis trichas"
    assert df.iloc[0].filename == "marsh.wav"            # stem resolved against audio_dir
    assert df.iloc[2].common_name == "CONI"              # unknown code: raw label kept as common name


def test_birdcode_filename_from_sibling_audio():
    df = dl.load_detections(FIX / "birdcode")
    assert set(df.filename) == {"20230730.wav", "20260623.flac"}
    assert df.scientific_name.tolist() == ["Corvus brachyrhynchos", "Corvus corax", "Catharus ustulatus"]


def test_perch_default_window():
    df = dl.load_detections(FIX / "perch_chirp")
    assert df.end_time.tolist() == [5.0, 10.0, 15.0]
    df = dl.load_detections(FIX / "perch_chirp", default_duration=3.0)
    assert df.end_time.tolist() == [3.0, 8.0, 13.0]      # explicit option overrides the 5 s default

def test_raven_dedups_views_and_infers_filename():
    df = dl.load_detections(FIX / "raven")
    assert df.filename.tolist() == ["rec1.wav", "rec1.wav"]
    assert df.label.tolist() == ["Great Tit", "Parus major"]
    assert df.scientific_name.tolist()[1] == "Parus major"
    assert df.confidence.isna().all()


def test_raven_multifile_offsets():
    df = dl.load_detections(FIX / "raven_multi")
    assert df.filename.tolist() == ["a.wav", "b.wav"]
    assert df.start_time.tolist() == [5.0, 5.0] and df.end_time.tolist() == [6.0, 6.5]


def test_kaleidoscope_id_csv():
    df = dl.load_detections(FIX / "kaleidoscope")
    r = df.iloc[0]
    assert r.filename.endswith("night1\\S4U0001_20240701_213000.wav")
    assert r.start_time == 0.0 and r.end_time == 5.0
    assert r.label == "PIPPIP" and r.confidence == pytest.approx(0.83)


def test_wide_table_min_confidence():
    df = dl.load_detections(FIX / "opensoundscape", min_confidence=0.5)
    assert df.label.tolist() == ["Turdus merula", "Erithacus rubecula"]
    assert df.confidence.tolist() == [0.91, 0.77]


def test_options_filename_mode_and_min_conf_and_extras():
    df = dl.load_detections(FIX / "birdnet_combined", filename_mode="basename", min_confidence=0.5)
    assert df.filename.tolist() == ["rec1.wav", "rec2.wav"]
    df = dl.load_detections(FIX / "birdnet_combined", filename_mode="stem")
    assert df.filename.tolist() == ["rec1", "rec1", "rec2"]
    df = dl.load_detections(FIX / "hawkears_raven", keep_extra_columns=True)
    assert "Low Freq (Hz)" in df.columns and "eBird Code" in df.columns


def test_generic_odd_headers_and_column_map(tmp_path):
    p = tmp_path / "weird.csv"
    p.write_text("rec,t_on,t_off,what,how_sure\nrec9.wav,1,2,Robin,0.5\n")
    with pytest.raises(ValueError):
        dl.load_detections(p)
    df = dl.load_detections(p, column_map={"rec": "filename", "t_on": "start_time", "t_off": "end_time",
                                           "what": "label", "how_sure": "confidence"})
    assert df.iloc[0].common_name == "Robin" and df.iloc[0].end_time == 2.0


def test_nested_json(tmp_path):
    p = tmp_path / "out.json"
    p.write_text('[{"file": "a.wav", "detections": [{"start": 0, "end": 1, "class": "Parus major", "score": 0.4}]},'
                 ' {"file": "b.wav", "detections": [{"start": 2, "end": 3, "class": "Wren", "score": 0.9}]}]')
    df = dl.load_detections(p)
    assert df.filename.tolist() == ["a.wav", "b.wav"] and df.label.tolist() == ["Parus major", "Wren"]


def test_excel_and_bom_and_utf16(tmp_path):
    x = tmp_path / "det.xlsx"
    pd.DataFrame({"File": ["a.wav"], "Start (s)": [1.0], "End (s)": [2.0], "Common name": ["Wren"],
                  "Scientific name": ["Troglodytes troglodytes"], "Confidence": [0.7]}).to_excel(x, index=False)
    assert dl.load_detections(x).iloc[0].scientific_name == "Troglodytes troglodytes"
    b = tmp_path / "bom.csv"
    b.write_bytes(b"\xef\xbb\xbfStart (s),End (s),Scientific name,Common name,Confidence,File\n0,3,Turdus merula,Eurasian Blackbird,0.5,r.wav\n")
    assert dl.load_detections(b).iloc[0].filename == "r.wav"
    u = tmp_path / "u16.txt"
    u.write_text("Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tSpecies\n1\tSpectrogram 1\t1\t1\t2\tWren\n", encoding="utf-16")
    assert dl.load_detections(u).iloc[0].label == "Wren"


def test_register_custom_format(tmp_path):
    @dl.register_format
    class MyToolCSV(dl.TableFormat):
        name = "mytool_csv"
        tool = "MyTool"
        required = frozenset({"clipname", "onset_sec", "offset_sec", "sp", "pr"})
        base_score = 90
        column_map = {"clipname": "filename", "onset_sec": "start_time", "offset_sec": "end_time",
                      "sp": "label", "pr": "confidence"}
    p = tmp_path / "mytool_out.csv"
    p.write_text("clipname,onset_sec,offset_sec,sp,pr\nz.wav,1,2,Corvus corax,0.9\n")
    df = dl.load_detections(p)
    assert df.attrs["source_format"] == "mytool_csv" and df.iloc[0].scientific_name == "Corvus corax"
    assert "mytool_csv" in [f.name for f in dl.available_formats()]


def test_split_label():
    assert dl.split_label("Turdus merula_Eurasian Blackbird") == ("Eurasian Blackbird", "Turdus merula")
    assert dl.split_label("Dog_Dog") == ("Dog", "Dog")
    assert dl.split_label("Corvus corax") == (None, "Corvus corax")
    assert dl.split_label("Corvus_corax") == (None, "Corvus corax")
    assert dl.split_label("Turdus merula, Eurasian Blackbird") == ("Eurasian Blackbird", "Turdus merula")
    assert dl.split_label("MAWR") == ("MAWR", None)
    assert dl.split_label("Great Tit") == ("Great Tit", None)
    assert dl.split_label("Engine") == ("Engine", None)
    assert dl.split_label(float("nan")) == (None, None)
