
# 1. Import libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display
from transformers import AutoProcessor, MusicgenForConditionalGeneration

# 2. Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

print("PyTorch:", torch.__version__)
print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# 3. Load pretrained MusicGen model
MODEL_ID = "facebook/musicgen-small"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)

model = model.to(device)
model.eval()

sample_rate = model.config.audio_encoder.sampling_rate

print("Model loaded:", MODEL_ID)
print("Sampling rate:", sample_rate, "Hz")


# 4. Define text prompt
prompt = (
    "Calm ambient piano with soft strings, slow tempo, "
    "peaceful evening atmosphere"
)

inputs = processor(
    text=[prompt],
    padding=True,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}


# 5. Generate music
torch.manual_seed(42)

if device == "cuda":
    torch.cuda.manual_seed_all(42)

with torch.no_grad():
    audio_values = model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=3.0,
        max_new_tokens=256
    )


# 6. Extract generated audio
audio_musicgen = audio_values[0, 0].detach().cpu().numpy()


# 7. Play generated music
display(Audio(audio_musicgen, rate=sample_rate))


# 8. Plot audio waveform
time = np.arange(len(audio_musicgen)) / sample_rate

plt.figure(figsize=(12, 4))
plt.plot(time, audio_musicgen)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title("MusicGen Output - Waveform")
plt.grid(alpha=0.25)
plt.show()