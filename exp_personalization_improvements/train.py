import os
import gc

import torch
import torch.nn.functional as F
import bitsandbytes as bnb

from torch.utils.data import DataLoader

from diffusers import StableDiffusion3Pipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file

from utils.custom_dataset import CustomDataset
from config.pipeline_config import PipelineConfig


def fine_tuning_lora(
    image_folder: str,
    output_dir: str,
    placeholder_token: str = 'sks',
    initializer_token: str = 'dog',
    instance_prompt: str = 'A photo of sks dog',
    base_model_id: str = 'stabilityai/stable-diffusion-3.5-medium',
    max_train_steps: int = 700,
    learning_rate: float = 1.5e-4,
    device: str = 'cuda'
):
    
    print('--- Fine-Tuning (using LoRA) Started ---\n')
    os.makedirs(output_dir, exist_ok=True)

    print('[T 1/9] Loading base model...')
    pipe = StableDiffusion3Pipeline.from_pretrained(
        base_model_id, 
        torch_dtype=torch.bfloat16
    ).to(device)

    # Register the placeholder token in all available tokenizers
    # and initialize its embedding from the class token
    print(f'[T 2/9] Registering placeholder token \'{placeholder_token}\' in tokenizers...')
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]

    for tokenizer, text_encoder in zip(tokenizers, text_encoders):
        if tokenizer is None or text_encoder is None:
            continue

        # Add the new placeholder token to the tokenizer
        num_added_tokens = tokenizer.add_tokens(placeholder_token)
        if num_added_tokens == 0:
            print(f'[INFO] Token \'{placeholder_token}\' already exists in the tokenizer.')
            continue

        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
        initializer_token_id = tokenizer.convert_tokens_to_ids(initializer_token)

        # Resize the text encoder's token embeddings to include the new token
        text_encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)

        # Initialize the placeholder token embedding by copying the embedding of the class/initializer token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

        # Freeze the text encoder to reduce GPU memory usage and keep its parameters fixed during LoRA training
        text_encoder.requires_grad_(False)


    # Dynamic Data Augmentation: prepare the dataset and data loader with data augmentation
    # Random transformations are applied dynamically when samples are loaded
    print('[T 3/9] Preparing CustomDataset and DataLoader with Data Augmentation...')
    dataset = CustomDataset(image_folder=image_folder, size=1024)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Keep the VAE on the GPU to encode randomly augmented crops into latent representations at each training step
    vae = pipe.vae.to(device)
    vae.eval()
    vae.requires_grad_(False)

    print('[T 4/9] Configuring LoRA on MM-DiT Transformer...')
    transformer = pipe.transformer.to(device)  

    transformer.enable_gradient_checkpointing()

    # Configure LoRA with a higher rank and additional target modules to adapt both the image and text-conditioning pathways
    lora_config = LoraConfig(
        r=32, 
        lora_alpha=32,
        target_modules=[
            'to_q', 'to_k', 'to_v', 'to_out.0',
            'add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out'
        ],
        init_lora_weights='gaussian'
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.train()

    print('[T 5/9] Applying AdamW 8-bit optimization (to save VRAM)...')
    try:
        optimizer = bnb.optim.AdamW8bit(transformer.parameters(), lr=learning_rate)
    except Exception:
        optimizer = torch.optim.AdamW(transformer.parameters(), lr=learning_rate)
        print('[WARN] Cannot apply AdamW 8-bit: falling back to standard AdamW')

    # Use a cosine learning rate schedule with a 10% warm-up period
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * 0.10),
        num_training_steps=max_train_steps
    )

    # Precompute the prompt embeddings using the updated placeholder token
    # The text encoders are only needed during this initial encoding step
    print('[T 6/9] Encoding text prompt with newly embedded token...')
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

    print('[T 7/9] Unloading Text Encoder from GPU (to save VRAM)...')
    del pipe.text_encoder, pipe.text_encoder_2
    if hasattr(pipe, 'text_encoder_3') and pipe.text_encoder_3 is not None:
        del pipe.text_encoder_3
    del pipe 

    gc.collect()
    torch.cuda.empty_cache()


    print(f'[T 8/9] Fine-tuning loop ({max_train_steps} step)...\n')
    global_step = 0
    data_iter = iter(dataloader)

    prompt_embeds = prompt_embeds.to(device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device)

    while global_step < max_train_steps:
        optimizer.zero_grad()

        # Fetch a randomly augmented image from the data loader
        # Restart the iterator when the end of the dataset is reached
        try:
            batch_images = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch_images = next(data_iter)

        batch_images = batch_images.to(device, dtype=torch.bfloat16)

        # Encode the augmented images into latent representations
        # The VAE remains frozen, so no gradients are computed here
        with torch.no_grad():
            latents = vae.encode(batch_images).latent_dist.sample()
            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor


        # Sample Gaussian noise and a random interpolation timestep
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
        lr_scheduler.step()

        global_step += 1
        if global_step % 50 == 0 or global_step == max_train_steps:
            current_lr = lr_scheduler.get_last_lr()[0]
            print(f'\tStep [{global_step}/{max_train_steps}] - Loss: {loss.item():.4f} - LR: {current_lr:.6f}')

    print(f'\n[T 9/9] Saving adaptation weights in: \'{output_dir}\' ...')
    peft_state_dict = get_peft_model_state_dict(transformer)

    sd3_lora_state_dict = {}
    for k, v in peft_state_dict.items():
        clean_key = k.replace('base_model.model.', '')
        sd3_lora_state_dict[f'transformer.{clean_key}'] = v

    save_path = os.path.join(output_dir, 'pytorch_lora_weights.safetensors')
    save_file(sd3_lora_state_dict, save_path)

    print('\n--- Fine-Tuning Ended! ---')

    del transformer, vae
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    if os.path.exists(PipelineConfig.data_dir):
        fine_tuning_lora(
            image_folder=PipelineConfig.data_dir,
            output_dir=PipelineConfig.adaptation_dir,
            placeholder_token=PipelineConfig.placeholder_token,
            initializer_token=PipelineConfig.class_token,
            instance_prompt=PipelineConfig.training_prompt,
            max_train_steps=700,
            learning_rate=1.5e-4
        )
    else:
        print(f'[ERROR] Please create folder \'{PipelineConfig.data_dir}\' and insert 5-10 images of the subject')
