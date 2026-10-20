
# 1. Environment setup
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import truststore
truststore.inject_into_ssl()

import urllib.request

with urllib.request.urlopen("https://huggingface.co", timeout=20) as r:
    print("HTTPS status:", r.status)


# 2. Import libraries
import torch
import matplotlib.pyplot as plt
from IPython.display import display
from diffusers import StableDiffusionPipeline


# 3. Select device
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("Device:", device)


# 4. Load pretrained Stable Diffusion model
MODEL_ID = "segmind/tiny-sd"

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype
)

pipe = pipe.to(device)

inference_steps = 20
guidance_scale = 7.5

print("Loaded:", MODEL_ID)


# 5. Generate a single image from a text prompt
prompt = (
    "Futuristic Mumbai skyline at sunset, cyberpunk skyscrapers "
    "beside iconic Indian architecture, neon Hindi and English signs, "
    "monsoon clouds, golden and magenta light, busy streets, "
    "futuristic vehicles, cinematic atmosphere, dramatic lighting, "
    "volumetric fog, photorealistic, ultra-detailed, wide-angle composition."
)

if not prompt.strip():
    raise ValueError("Enter a text prompt.")

seed = 123
generator = torch.Generator(device=device).manual_seed(seed)

image = pipe(
    prompt=prompt,
    num_inference_steps=inference_steps,
    guidance_scale=guidance_scale,
    generator=generator
).images[0]

display(image)

image.save("generated_image.png")
print("Image saved as generated_image.png")


# 6. Generate images using different random seeds
seeds = [10, 20, 30]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, s in zip(axes, seeds):
    generator = torch.Generator(device=device).manual_seed(s)

    img = pipe(
        prompt=prompt,
        num_inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]

    ax.imshow(img)
    ax.set_title(f"Seed {s}")
    ax.axis("off")

plt.tight_layout()
plt.show()


# 7. Compare simple and detailed prompts
simple_prompt = "A futuristic Mumbai skyline at sunset."

detailed_prompt = (
    "A futuristic Mumbai skyline at sunset, towering cyberpunk "
    "skyscrapers rising beside iconic Indian architecture, "
    "futuristic vehicles moving through busy streets, people walking "
    "beneath glowing Hindi and English neon signs, monsoon clouds "
    "covering the sky, golden and magenta sunlight reflecting on "
    "glass buildings and wet roads, cinematic cyberpunk style, "
    "dramatic lighting, volumetric fog, photorealistic details, "
    "wide-angle composition."
)

if not simple_prompt.strip() or not detailed_prompt.strip():
    raise ValueError("Enter both prompts.")

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

for ax, text, title in zip(
    axes,
    [simple_prompt, detailed_prompt],
    ["Simple Prompt", "Detailed Prompt"]
):
    generator = torch.Generator(device=device).manual_seed(55)

    img = pipe(
        prompt=text,
        num_inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]

    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()