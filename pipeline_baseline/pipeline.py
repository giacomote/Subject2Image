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

from pipeline_baseline.custom_dataset import CustomDataset


class BaselinePipe:
    def __init__(self):
            self._model_id = 'stabilityai/stable-diffusion-3.5-large'
            self._weight_name = 'base_lora_weights.safetensors'
            self._inference_pipe = None
    
    def _free_inference_memory(self):
        if self._inference_pipe is not None:
            del self._inference_pipe
            self._inference_pipe = None
            gc.collect()
            torch.cuda.empty_cache()

    def fine_tuning_lora(
        self,
        image_folder: str,
        output_dir: str,
        instance_prompt: str = 'A photo of <sks> dog',
        max_train_steps: int = 1200,
        learning_rate: float = 1e-4,
        device: str = 'cuda'
    ):

        self._free_inference_memory()
        
        print('--- Fine-Tuning (using LoRA) Started ---\n')
        os.makedirs(output_dir, exist_ok=True)

        print('[T 1/9] Loading base model...')
        pipe = StableDiffusion3Pipeline.from_pretrained(
            self._model_id, 
            dtype=torch.bfloat16
        ).to(device)

        print('[T 2/9] Computing embeddings for prompt...')
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

        print('[T 3/9] Unloading Text Encoder from GPU (to save VRAM)...')
        del pipe.text_encoder, pipe.text_encoder_2
        if hasattr(pipe, 'text_encoder_3') and pipe.text_encoder_3 is not None:
            del pipe.text_encoder_3

        gc.collect()
        torch.cuda.empty_cache()

        print('[T 4/9] Encoding images (in latent space)...')
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

        print('[T 5/9] Unloading VAE and Pipe from GPU (to save VRAM)...')
        transformer = pipe.transformer.to(device)  # Saving transformer (only thing to keep after pipeline deletion)

        del pipe
        gc.collect()
        torch.cuda.empty_cache()

        print('[T 6/9] Configuring LoRA on MM-DiT...')
        transformer.enable_gradient_checkpointing()
        
        lora_config = LoraConfig(
            r=32, 
            lora_alpha=32,
            target_modules=['to_q', 'to_k', 'to_v', 'to_out.0', 'add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out'],
            init_lora_weights='gaussian'
        )
        transformer = get_peft_model(transformer, lora_config)
        transformer.train()

        print('[T 7/9] Applying AdamW 8-bit optimization (to save VRAM)...')
        try:
            optimizer = bnb.optim.AdamW8bit(transformer.parameters(), lr=learning_rate)
        except Exception:
            optimizer = torch.optim.AdamW(transformer.parameters(), lr=learning_rate)
            print('[WARN] Cannot apply AdamW 8-bit: falling back to standard AdamW')

        # LR Scheduler
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(max_train_steps * 0.10),  # Warm-up over the first 10% of the steps
            num_training_steps=max_train_steps
        )

        print(f'[T 8/9] Fine-tuning loop ({max_train_steps} step)...\n')
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

        save_path = os.path.join(output_dir, self._weight_name)
        save_file(sd3_lora_state_dict, save_path)

        print('\n--- Fine-Tuning Ended! ---')

        del transformer
        gc.collect()
        torch.cuda.empty_cache()

    def generate_personalized_image(
        self,
        lora_dir: str,
        prompt: str,
        output_filename: str = 'result.png',
        device: str = 'cuda'
    ):
        
        print('--- Personalized Image Generation Started ---\n')
        
        if self._inference_pipe is None:
            print('[I 1/4] Loading base model for inference...')
            self._inference_pipe = StableDiffusion3Pipeline.from_pretrained(
                self._model_id,
                dtype=torch.bfloat16
            ).to(device)

            # pipe.enable_model_cpu_offload()  # Automatic offloading to GPU (to avoid using only the GPU)
            # pipe.vae.enable_tiling()  # Tiling for VAE decoding (to save VRAM)

            print(f'[I 2/4] Loading adaptation weights from \'{lora_dir}\' ...')
            self._inference_pipe.load_lora_weights(lora_dir, weight_name=self._weight_name)

        else:
            print('[I 1/4] Reusing cached base model...')
            print('[I 2/4] Reusing cached adaptation weights...')

        print(f'[I 3/4] Generating image...')
        image = self._inference_pipe(
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

        print('\n--- Personalized Image Generation Ended! ---')