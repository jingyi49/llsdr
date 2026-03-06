from audiotools import AudioSignal
import sys
import os
from tqdm import tqdm
import torchaudio
import torch
sys.path.insert(0, '.')
import vodac
model_path = "./runs/whole/best/vodac/weights.pth"
model = vodac.VODAC.load(model_path).cuda()
print(f"Model size: {sum([x.numel() for x in model.state_dict().values()]) / 1e6:.2f}M") # 74.26M
model.eval()
sample_rate = 16000
scp_file = "your_path_to/sim_reverb.list"
output_dir = "your_path_to_output_dir"
os.makedirs(output_dir, exist_ok=True)
with open(scp_file, "r") as f:
    wav_paths = [line.strip() for line in f.readlines() if line.strip().endswith(".wav")]

print(f"find {len(wav_paths)} audio files.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for idx, wav_path in tqdm(enumerate(wav_paths, 1), total=len(wav_paths)):
        # step one load audio files
    audio_input, sr = torchaudio.load(wav_path)
    if sr != sample_rate:
        print(f"sr:{sr}")
        audio_input = torchaudio.functional.resample(audio_input, sr, sample_rate)
    audio_input = audio_input.mean(dim=0, keepdim=True).to(device)  # Convert to mono
    audio_input = audio_input.unsqueeze(1)  # Remove channel dimension
    with torch.inference_mode():
        x = model.preprocess(audio_input, sample_rate)
        features, z, codes, _, _, _ = model.encode(x)
        enhanced_audio = model.decode(z)
    output_path = os.path.join(output_dir, os.path.basename(wav_path))
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    if enhanced_audio.dim() == 1:
        enhanced_audio = enhanced_audio.unsqueeze(0)
    if enhanced_audio.dim() == 3:
        enhanced_audio = enhanced_audio.squeeze(1)
    torchaudio.save(output_path, enhanced_audio.cpu(), sample_rate)
    print(f"[{idx}/{len(wav_paths)}] save complete: {output_path}")