"""Create sample output files that mirror what each tool really writes (from reading their source)."""
from pathlib import Path
import pandas as pd

root = Path(__file__).parent / "fixtures"
root.mkdir(exist_ok=True)

def w(rel, text):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p

# ---------------- BirdNET-Analyzer (analyze/core.py save_as_* ) ----------------
# combined csv
w("birdnet_combined/BirdNET_CombinedTable.csv",
  "Start (s),End (s),Scientific name,Common name,Confidence,File\n"
  "0.0,3.0,Turdus merula,Eurasian Blackbird,0.8123,/data/site1/rec1.wav\n"
  "3.0,6.0,Erithacus rubecula,European Robin,0.4411,/data/site1/rec1.wav\n"
  "0.0,3.0,Cyanistes caeruleus,Eurasian Blue Tit,0.9101,/data/site1/rec2.wav\n")
# analysis params file sits next to it and should be ignored
w("birdnet_combined/birdnet.analyze-params.csv", "Parameter,Value\nModel,BirdNET_GLOBAL_6K_V2.4\n")
# audacity combined
w("birdnet_combined/BirdNET_AudacityLabels.txt",
  "0.0\t3.0\tTurdus merula_Eurasian Blackbird\t0.8123\n"
  "3.0\t6.0\tErithacus rubecula_European Robin\t0.4411\n")
# per-file (split tables) outputs
w("birdnet_split/rec1.BirdNET.results.csv",
  "Start (s),End (s),Scientific name,Common name,Confidence,File\n"
  "0.0,3.0,Turdus merula,Eurasian Blackbird,0.8123,/data/site1/rec1.wav\n")
w("birdnet_split/rec2.BirdNET.results.csv",
  "Start (s),End (s),Scientific name,Common name,Confidence,File\n"
  "0.0,3.0,Cyanistes caeruleus,Eurasian Blue Tit,0.9101,/data/site1/rec2.wav\n")
w("birdnet_split_raven/rec1.BirdNET.selection.table.txt",
  "Selection\tBegin Time (s)\tEnd Time (s)\tCommon Name\tScientific Name\tSpecies Code\tConfidence\tView\tChannel\tFile Offset (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tBegin Path\n"
  "1\t0.0\t3.0\tEurasian Blackbird\tTurdus merula\teurbla\t0.8123\tSpectrogram 1\t1\t0.0\t0\t15000\t/data/site1/rec1.wav\n"
  "2\t3.0\t6.0\tEuropean Robin\tErithacus rubecula\teurrob1\t0.4411\tSpectrogram 1\t1\t3.0\t0\t15000\t/data/site1/rec1.wav\n")
w("birdnet_split_audacity/rec1.BirdNET.results.txt",
  "0.0\t3.0\tTurdus merula_Eurasian Blackbird\t0.8123\n3.0\t6.0\tErithacus rubecula_European Robin\t0.4411\n")
w("birdnet_split_audacity/rec2.BirdNET.results.txt",
  "0.0\t3.0\tCyanistes caeruleus_Eurasian Blue Tit\t0.9101\n")
# combined raven table with File Offset (Begin Time accumulates across files)
w("birdnet_raven_combined/BirdNET_SelectionTable.txt",
  "Selection\tBegin Time (s)\tEnd Time (s)\tCommon Name\tScientific Name\tSpecies Code\tConfidence\tView\tChannel\tFile Offset (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tBegin Path\n"
  "1\t0.0\t3.0\tEurasian Blackbird\tTurdus merula\teurbla\t0.8123\tSpectrogram 1\t1\t0.0\t0\t15000\t/data/site1/rec1.wav\n"
  "2\t3.0\t6.0\tEuropean Robin\tErithacus rubecula\teurrob1\t0.4411\tSpectrogram 1\t1\t3.0\t0\t15000\t/data/site1/rec1.wav\n"
  "3\t60.0\t63.0\tEurasian Blue Tit\tCyanistes caeruleus\teubtit\t0.9101\tSpectrogram 1\t1\t0.0\t0\t15000\t/data/site1/rec2.wav\n")
# kaleidoscope
w("birdnet_kaleidoscope/BirdNET_Kaleidoscope.csv",
  "INDIR,FOLDER,IN FILE,OFFSET,DURATION,scientific_name,TOP1MATCH,TOP1DIST\n"
  "/data,site1,rec1.wav,0.0,3.0,Turdus merula,Eurasian Blackbird,0.8123\n"
  "/data,site1,rec1.wav,3.0,3.0,Erithacus rubecula,European Robin,0.4411\n")
# parquet
pd.DataFrame({"Start (s)": [0.0, 3.0], "End (s)": [3.0, 6.0],
              "Scientific name": ["Turdus merula", "Erithacus rubecula"],
              "Common name": ["Eurasian Blackbird", "European Robin"],
              "Confidence": [0.8123, 0.4411], "File": ["/data/site1/rec1.wav"] * 2}
             ).to_parquet(root / "birdnet_parquet" / "BirdNET_CombinedTable.parquet", index=False) \
    if (root / "birdnet_parquet").mkdir(exist_ok=True) is None else None
# old birdnet "r" table
w("birdnet_rtable/BirdNET_RTable.csv",
  "filepath,start,end,scientific_name,common_name,confidence,lat,lon,week,overlap,sensitivity,min_conf,species_list,model\n"
  "/data/rec1.wav,0.0,3.0,Turdus merula,Eurasian Blackbird,0.8123,-1,-1,-1,0.0,1.0,0.1,,BirdNET_GLOBAL_6K_V2.4_Model_FP32\n")

# ---------------- HawkEars 2.x ----------------
w("hawkears_csv/scores.csv",
  "recording,name,start_time,end_time,score\n"
  "marsh,COYE,1.250,4.250,0.912\nmarsh,MAWR,2.000,4.000,0.880\nnight,CONI,1.250,4.500,0.910\n")
w("hawkears_csv/rarities.csv", "recording,name,start_time,end_time,score\nmarsh,KIRA,10.000,12.000,0.750\n")
w("hawkears_audacity/marsh_scores.txt", "2\t4\tMAWR;0.88\n5.5\t8\tCOYE;0.91\n")
w("hawkears_audacity/night_scores.txt", "1.25\t4.5\tCONI;0.75\n")
w("hawkears_audacity/quiet_scores.txt", "")            # recording with no detections
w("hawkears_audacity/rarities/marsh_scores.txt", "3\t6\tKIRA;0.7\n")
w("hawkears_audacity/marsh.wav", "")                   # pretend audio next to the labels
w("hawkears_audacity/night.mp3", "")
w("hawkears_audacity_names/marsh_scores.txt", "2.5\t5.75\tCistothorus palustris;0.876\n")
w("hawkears_raven/marsh.HawkEars.selection.table.txt",
  "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tCommon Name\tScientific Name\tSpecies Code\teBird Code\tSpecies\tConfidence\tFile Offset (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tBegin File\tBegin Path\n"
  "1\tSpectrogram 1\t1\t1.25\t4.25\tCommon Yellowthroat\tGeothlypis trichas\tCOYE\tcomyel\tCOYE\t0.912345\t1.25\t500\t8000\tmarsh.wav\t/data/marsh.wav\n")

# ---------------- BirdCODE ----------------
w("birdcode/audio/BirdCODE_predictions/20230730.txt",
  "Begin Time (s)\tEnd Time (s)\tSpecies\tScore\n3.8157894736842106\t5.7894736842105265\tCorvus brachyrhynchos\t0.6134416\n")
w("birdcode/audio/BirdCODE_predictions/20260623.txt",
  "Begin Time (s)\tEnd Time (s)\tSpecies\tScore\n"
  "0.2631578947368421\t1.5789473684210527\tCorvus corax\t0.8774136\n"
  "3.68421052631579\t4.078947368421053\tCatharus ustulatus\t0.57417274\n")
w("birdcode/audio/20230730.wav", ""); w("birdcode/audio/20260623.flac", "")

# ---------------- Perch ----------------
w("perch_chirp/inference.csv",
  "filename, timestamp_s, label, logit\n"           # note: chirp writes ', '-joined header
  "site1/rec1.wav,0.00,comrav,1.23\nsite1/rec1.wav,5.00,comrav,-0.40\nsite1/rec2.wav,10.00,amecro,2.10\n")
w("perch_hoplite/detections.csv",
  "idx,project,filename,window_start,window_end,label,logits\n"
  "17,demo,rec1.wav,0.0,5.0,comrav,1.23\n18,demo,rec1.wav,5.0,10.0,amecro,0.77\n")

# ---------------- Raven Pro ----------------
w("raven/rec1.Table.1.selections.txt",
  "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tDelta Time (s)\tSpecies\tNotes\n"
  "1\tSpectrogram 1\t1\t12.5\t14.0\t2000\t6000\t1.5\tGreat Tit\tclear\n"
  "1\tWaveform 1\t1\t12.5\t14.0\t\t\t1.5\tGreat Tit\tclear\n"
  "2\tSpectrogram 1\t1\t30.1\t31.0\t1000\t4000\t0.9\tParus major\t\n")
w("raven/rec1.wav", "")
# multi-file Raven table (opened as a sequence of files)
w("raven_multi/deployment.Table.1.selections.txt",
  "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tBegin File\tFile Offset (s)\tAnnotation\n"
  "1\tSpectrogram 1\t1\t5.0\t6.0\t1000\t3000\ta.wav\t5.0\tEuropean Robin\n"
  "2\tSpectrogram 1\t1\t65.0\t66.5\t1000\t3000\tb.wav\t5.0\tWren\n")

# ---------------- Kaleidoscope ----------------
w("kaleidoscope/id.csv",
  "INDIR,FOLDER,IN FILE,CHANNEL,OFFSET,DURATION,OUT FILE FS,OUT FILE ZC,DATE,TIME,HOUR,DATE-12,TIME-12,HOUR-12,AUTO ID*,PULSES,MATCHING,MATCH RATIO,MARGIN,ALTERNATE 1,ALTERNATE 2,N,Fc,Sc,Dur,Fmax,Fmin,Fmean,TBC,Fk,Tk,S1,Tc,Qual,FILES,MANUAL ID,ORGID,USERID,REVIEW ORGID,REVIEW USERID,INPATHMD5,OUTPATHMD5FS,OUTPATHMD5ZC\n"
  "D:\\bats,night1,S4U0001_20240701_213000.wav,0,0,5,,,2024-07-01,21:30:00,21,,,,PIPPIP,12,10,0.83,0.5,PIPPYG,,,45.1,,,,,,,,,,,,,,,,,,,,,\n")

# ---------------- OpenSoundscape wide table ----------------
w("opensoundscape/scores.csv",
  "file,start_time,end_time,Turdus merula,Erithacus rubecula,Parus major\n"
  "rec1.wav,0.0,3.0,0.91,0.05,0.10\nrec1.wav,3.0,6.0,0.20,0.77,0.02\n")

# ---------------- generic / unknown ----------------
w("generic_json/detections.json",
  '[{"audio_file": "x.wav", "onset": 1.0, "offset_time": 2.5, "species": "Anthus trivialis", "prob": 0.66}]')
w("generic_semicolon/output.csv",
  "File;Start (sec);End (sec);Class;Probability\nrec.wav;1,5;2,5;Blue Tit;0,42\n")
w("generic_clock/table.tsv", "clip\tt_start\tt_end\tprediction\tp\nrec.wav\t00:01:05.5\t00:01:07\tWren\t0.9\n")
w("generic_headers_only/empty.csv", "Start (s),End (s),Scientific name,Common name,Confidence,File\n")
# not detections at all
w("noise/notes.txt", "Just some field notes.\nNothing tabular here.\n")
w("noise/random.csv", "a,b\n1,2\n")
print("fixtures written to", root)
