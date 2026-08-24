import os
import gc

import torch
from diffusers import StableDiffusion3Pipeline

from config.pipeline_config import PipelineConfig


def generate_personalized_image(
    lora_dir: str,
    prompt: str,
    output_filename: str = 'result.png',
    base_model_id: str = 'stabilityai/stable-diffusion-3.5-large'
):
    
    print('--- Personalized Image Generation Started ---\n')
    
    print('[I 1/4] Loading base model...')
    pipe = StableDiffusion3Pipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()  # Automatic offloading to GPU (to avoid using only the GPU)

    pipe.vae.enable_tiling()  # Tiling for VAE decoding (to save VRAM)

    print(f'[I 2/4] Loading adaptation weights from \'{lora_dir}\' ...')
    pipe.load_lora_weights(lora_dir)

    print(f'[I 3/4] Generating image...')
    image = pipe(
        prompt=prompt,
        negative_prompt='blurry, distorted, low quality, bad anatomy',
        num_inference_steps=28,
        guidance_scale=4.5,
        width=1024,
        height=1024
    ).images[0]

    print('[I 4/4] Saving result...')
    image.save(output_filename)
    print(f'[OK] Image saved: {output_filename}')

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    print('\n--- Personalized Image Generation Ended!---')


if __name__ == '__main__':
    os.makedirs(PipelineConfig.results_dir, exist_ok=True)

    output_file = PipelineConfig.results_dir + str(len(os.listdir(PipelineConfig.results_dir))) + '.png'

    generate_personalized_image(
        lora_dir=PipelineConfig.adaptation_dir,
        prompt=PipelineConfig.generation_prompt,
        output_filename=output_file
    )