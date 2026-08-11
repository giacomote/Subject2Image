import os

import torch
from diffusers import StableDiffusion3Pipeline

from config.pipeline_config import PipelineConfig


def generate_personalized_image(
    lora_dir: str,
    prompt: str,
    output_filename: str = 'result.png',
    base_model_id: str = 'stabilityai/stable-diffusion-3.5-medium',
    device: str = 'cuda'
):
    print('--- Personalized Image Generation Started ---\n')
    
    print('[1/4] Loading base model...')
    pipe = StableDiffusion3Pipeline.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16
    ).to(device)

    print(f'[2/4] Loading adaptation weights from \'{lora_dir}\' ...')
    pipe.load_lora_weights(lora_dir)

    print(f'[3/4] Generating image...')
    image = pipe(
        prompt=prompt,
        negative_prompt='blurry, distorted, low quality, bad anatomy',
        num_inference_steps=28,
        guidance_scale=4.5,
        width=1024,
        height=1024
    ).images[0]

    print('[4/4] Saving result...')
    image.save(output_filename)
    print(f'[OK] Image saved: {output_filename}')

    print('\n--- Personalized Image Generation Ended!---')


if __name__ == '__main__':
    LORA_OUTPUT_DIR = PipelineConfig.adaptation_dir
    TOKEN_IDENTIFIER = 'sks cat'

    os.makedirs(PipelineConfig.results_dir, exist_ok=True)
    
    TEST_PROMPT = f'A high quality studio photograph of {TOKEN_IDENTIFIER} on a table, with the background of a kitchen'
    OUTPUT_FILE = PipelineConfig.results_dir + str(len(os.listdir(PipelineConfig.results_dir))) + '.png'

    generate_personalized_image(
        lora_dir=LORA_OUTPUT_DIR,
        prompt=TEST_PROMPT,
        output_filename=OUTPUT_FILE
    )