import os
import gc

import torch
import torch.nn.functional as F
import bitsandbytes as bnb

from torch.utils.data import DataLoader

from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file

from utils.custom_dataset import CustomDataset
from config.pipeline_config import PipelineConfig


def train_dreambooth_lora(
    image_folder: str,
    output_dir: str,
    instance_prompt: str = 'a photo of sks cat',
    base_model_id: str = 'stabilityai/stable-diffusion-3.5-medium',
    max_train_steps: int = 400,
    learning_rate: float = 1e-4,
    device: str = 'cuda'
):
    print('--- Fine-Tuning (using LoRA) Started ---\n')
    os.makedirs(output_dir, exist_ok=True)

    print('[1/9] Loading base model...')
    pipe = StableDiffusion3Pipeline.from_pretrained(
        base_model_id, 
        torch_dtype=torch.bfloat16
    ).to(device)

    print('[2/9] Computing embeddings for prompt...')
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = (
            pipe.encode_prompt(
                prompt=instance_prompt,
                prompt_2=instance_prompt,
                prompt_3=instance_prompt,
                do_classifier_free_guidance=False,
                device=device
            )
        )

    print('[3/9] Unloading Text Encoder from GPU (to save VRAM)...')
    del pipe.text_encoder, pipe.text_encoder_2
    if hasattr(pipe, 'text_encoder_3') and pipe.text_encoder_3 is not None:
        del pipe.text_encoder_3

    gc.collect()
    torch.cuda.empty_cache()

    print('[4/9] Encoding images (in latent space)...')
    dataset = CustomDataset(image_folder=image_folder, size=1024)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    cached_latents = []
    pipe.vae.to(device)
    pipe.vae.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device, dtype=torch.bfloat16)
            lat = pipe.vae.encode(batch).latent_dist.sample()
            lat = (lat - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            cached_latents.append(lat.cpu())  # Temporary saving latents in RAM

    print('[5/9] Unloading VAE and Pipe from GPU (to save VRAM)...')
    transformer = pipe.transformer.to(device)  # Saving transformer (only thing to keep after pipeline deletion)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    print('[6/9] Configuring LoRA on MM-DiT...')
    transformer.enable_gradient_checkpointing()
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=['to_q', 'to_k', 'to_v', 'to_out.0'],
        init_lora_weights='gaussian'
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.train()

    print('[7/9] Applying AdamW 8-bit optimization (to save VRAM)...')
    try:
        optimizer = bnb.optim.AdamW8bit(transformer.parameters(), lr=learning_rate)
    except Exception:
        optimizer = torch.optim.AdamW(transformer.parameters(), lr=learning_rate)
        print('[WARN] Cannot apply AdamW 8-bit: falling back to standard AdamW')

    print(f'[8/9] Fine-tuning loop ({max_train_steps} step)...\n')
    global_step = 0
    num_samples = len(cached_latents)

    prompt_embeds = prompt_embeds.to(device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device)

    while global_step < max_train_steps:
        optimizer.zero_grad()
        
        latent_idx = global_step % num_samples
        latents = cached_latents[latent_idx].to(device, dtype=torch.bfloat16)

        noise = torch.randn_like(latents, dtype=torch.float32).to(dtype=torch.bfloat16)
        u = torch.rand((latents.shape[0],), device=device, dtype=torch.bfloat16)
        u_expanded = u.view(-1, 1, 1, 1)
        
        noisy_latents = (1.0 - u_expanded) * latents + u_expanded * noise
        target = noise - latents
        timesteps = (u * 1000.0).to(dtype=torch.bfloat16)

        model_pred = transformer(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False
        )[0]

        loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')
        loss.backward()

        optimizer.step()

        global_step += 1
        if global_step % 50 == 0 or global_step == max_train_steps:
            print(f'\tStep [{global_step}/{max_train_steps}] - Loss: {loss.item():.4f}')

    print(f'\n[9/9] Saving adaptation weights in: \'{output_dir}\' ...')
    peft_state_dict = get_peft_model_state_dict(transformer)

    sd3_lora_state_dict = {}
    for k, v in peft_state_dict.items():
        clean_key = k.replace('base_model.model.', '')
        sd3_lora_state_dict[f'transformer.{clean_key}'] = v

    save_path = os.path.join(output_dir, 'pytorch_lora_weights.safetensors')
    save_file(sd3_lora_state_dict, save_path)

    print('\n--- Fine-Tuning Ended! ---')

    del transformer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    IMAGES_DIR = PipelineConfig.data_dir
    LORA_OUTPUT_DIR = PipelineConfig.adaptation_dir
    TOKEN_IDENTIFIER = 'sks cat'
    INSTANCE_PROMPT = f'a photo of {TOKEN_IDENTIFIER}'

    if os.path.exists(IMAGES_DIR):
        train_dreambooth_lora(
            image_folder=IMAGES_DIR,
            output_dir=LORA_OUTPUT_DIR,
            instance_prompt=INSTANCE_PROMPT,
            max_train_steps=400,
            learning_rate=1e-4
        )
    else:
        print(f'[ERROR] Please create folder \'{IMAGES_DIR}\' and insert 5-10 images of the subject')
