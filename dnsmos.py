import os
import csv
import torchaudio.transforms as T
import torchaudio
import torch
from speechmos import dnsmos
from tqdm import tqdm
import numpy as np


folder = "your_path_to_the_output_dir"
target_sr = 16000  # DNSMOS need 16kHz
output_csv = "result_dnsmos.csv"
batch_size = 1


wav_files = [
    os.path.join(folder, f)
    for f in sorted(os.listdir(folder))
    if f.endswith(".wav")
]
if not wav_files:
    raise FileNotFoundError(f"❌ do not find any files in {folder}")

fieldnames = ["filenames", "ovrl_mos", "sig_mos", "bak_mos", "p808_mos"]
rows = []

for i in tqdm(range(0, len(wav_files), batch_size), desc="Computing DNSMOS"):
    wav_file = wav_files[i]
    try:
        wav, sr = torchaudio.load(wav_file)
        if sr != target_sr:
            wav = T.Resample(sr, target_sr)(wav)
        audio = wav.mean(dim=0, keepdim=True)  # single channel
        audio_np = audio.squeeze(0).cpu().numpy()

        # calculate DNSMOS
        result_dict = dnsmos.run(audio_np, sr=target_sr)
        rows.append({
            "filenames": ",".join(os.path.basename(wav_file)),
            **{k: float(v) for k, v in result_dict.items()}
        })
    except Exception as e:
        print(f"⚠️  file {wav_file}: {e}")

# ========== save as CSV ==========
with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# ========== mean score ==========
avg_scores = {k: sum(r[k] for r in rows) / len(rows) for k in fieldnames[1:]}
print("\n==== 平均分数 ====")
for k, v in avg_scores.items():
    print(f"{k}: {v:.4f}")
print(f"\n结果 for {folder}已保存到: {os.path.abspath(output_csv)}")

