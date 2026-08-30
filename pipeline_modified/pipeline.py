import os
import gc

import torch
import torch.nn.functional as F
import bitsandbytes as bnb

from torch.utils.data import DataLoader

from diffusers import StableDiffusion3Pipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file, load_file

from pipeline_modified.custom_dataset import CustomDataset

class ModifiedPipe:
    def __init__(self):
        self._model_id = 'stabilityai/stable-diffusion-3.5-large'
        self._lora_weight_name = 'mod_lora_weights.safetensors'
        self._embeds_weight_name = 'mod_learned_embeds.safetensors'
        self._inference_pipe = None

    def _free_inference_memory(self):
        if self._inference_pipe is not None:
            del self._inference_pipe
            self._inference_pipe = None
            gc.collect()
            torch.cuda.empty_cache()

    def fine_tuning_lora_with_textual_inversion(
        self,
        image_folder: str,
        output_dir: str,
        placeholder_token: str = 'sks',
        initializer_token: str = 'dog',
        instance_prompt: str = 'A photo of sks dog',
        max_train_steps: int = 1200,
        lora_learning_rate: float = 5e-5,
        ti_learning_rate: float = 1e-4,
        ti_text_encoders: list[int] = [1, 2],
        device: str = 'cuda'
    ):
        self._free_inference_memory()

        print('--- Fine-Tuning (Textual Inversion + LoRA) Started ---\n')
        print(f'[INFO] Training Textual Inversion on text encoders: {ti_text_encoders}\n')
        os.makedirs(output_dir, exist_ok=True)

        print('[T 1/9] Loading base model...')
        pipe = StableDiffusion3Pipeline.from_pretrained(
            self._model_id,
            torch_dtype=torch.bfloat16
        ).to(device)

        print(f'[T 2/9] Registering placeholder token \'{placeholder_token}\' in tokenizers and text encoders...')
        tokenizers = [pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3]
        text_encoders = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]
        placeholder_token_ids = []

        for idx, (tokenizer, text_encoder) in enumerate(zip(tokenizers, text_encoders), start=1):
            if tokenizer is None or text_encoder is None:
                placeholder_token_ids.append(None)
                continue

            num_added_tokens = tokenizer.add_tokens(placeholder_token)
            assert num_added_tokens == 1, (f'Placeholder token \'{placeholder_token}\' was not added as exactly one token.')

            placeholder_id = tokenizer.convert_tokens_to_ids(placeholder_token)
            placeholder_token_ids.append(placeholder_id)

            # Resize embedding table
            text_encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)

            # Copy initializer token weights
            initializer_id = tokenizer.convert_tokens_to_ids(initializer_token)
            token_embeds = text_encoder.get_input_embeddings().weight.data
            token_embeds[placeholder_id] = token_embeds[initializer_id]

            # Freeze text encoder except input embeddings (only if enabled in ti_text_encoders)
            text_encoder.requires_grad_(False)
            if idx in ti_text_encoders:
                text_encoder.get_input_embeddings().requires_grad_(True)

        print('[T 3/9] Encoding images (in latent space using VAE)...')
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
                cached_latents.append(lat.cpu())

        print('[T 4/9] Unloading VAE from GPU to save VRAM...')
        del pipe.vae
        gc.collect()
        torch.cuda.empty_cache()

        print('[T 5/9] Configuring LoRA on MM-DiT Transformer...')
        transformer = pipe.transformer.to(device)
        transformer.enable_gradient_checkpointing()

        lora_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=['to_q', 'to_k', 'to_v', 'to_out.0', 'add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out'],
            init_lora_weights='gaussian'
        )

        transformer = get_peft_model(transformer, lora_config)
        transformer.train()

        # Collect parameters: LoRA weights + Active Text Encoder embeddings
        lora_params = [param for param in transformer.parameters() if param.requires_grad]
        ti_params = []

        for idx, text_encoder in enumerate(text_encoders, start=1):
            if text_encoder is not None and idx in ti_text_encoders:
                ti_params.extend(list(text_encoder.get_input_embeddings().parameters()))

        print('[T 6/9] Applying AdamW 8-bit optimization (to save VRAM)...')
        try:
            optimizer = bnb.optim.AdamW8bit(
                [{'params': lora_params, 'lr': lora_learning_rate},
                 {'params': ti_params, 'lr': ti_learning_rate}
                ])
        except Exception:
            optimizer = torch.optim.AdamW(
                [{'params': lora_params, 'lr': lora_learning_rate},
                 {'params': ti_params, 'lr': ti_learning_rate}
                ])
            print('[WARN] Cannot apply AdamW 8-bit: falling back to standard AdamW')

    # LR Scheduler
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(max_train_steps * 0.10),  # Warm-up over the first 10% of the steps
            num_training_steps=max_train_steps
        )

        print(f'[T 7/9] Fine-tuning loop ({max_train_steps} steps)...')
        global_step = 0
        num_samples = len(cached_latents)

        while global_step < max_train_steps:
            optimizer.zero_grad()

            prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=instance_prompt,
                prompt_2=instance_prompt,
                prompt_3=instance_prompt,
                do_classifier_free_guidance=False,
                device=device
            )

            latent_idx = global_step % num_samples
            latents = cached_latents[latent_idx].to(device, dtype=torch.bfloat16)

            noise = torch.randn_like(latents, dtype=torch.float32).to(dtype=torch.bfloat16)
            u = torch.rand((latents.shape[0],), device=device, dtype=torch.bfloat16)
            u_expanded = u.view(-1, 1, 1, 1)

            noisy_latents = ((1.0 - u_expanded) * latents + u_expanded * noise)
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

            # Zero out gradients for all tokens except the placeholder token (only for active text encoders)
            for idx, (text_encoder, ph_id) in enumerate(zip(text_encoders, placeholder_token_ids), start=1):
                if text_encoder is not None and ph_id is not None and idx in ti_text_encoders:
                    grads = text_encoder.get_input_embeddings().weight.grad
                    if grads is not None:
                        index_grads_to_zero = torch.ones(grads.shape[0], dtype=torch.bool, device=grads.device)
                        index_grads_to_zero[ph_id] = False
                        grads[index_grads_to_zero] = 0

            optimizer.step()
            lr_scheduler.step()

            global_step += 1
            if global_step % 50 == 0 or global_step == max_train_steps:
                current_lrs = lr_scheduler.get_last_lr()
                print(f'\tStep [{global_step}/{max_train_steps}] - Loss: {loss.item():.4f} - LoRA LR: {current_lrs[0]:.6f} - TI LR: {current_lrs[1]:.6f}')

        print(f'\n[T 8/9] Saving adaptation weights and learned embeddings to: \'{output_dir}\' ...')

        # Save LoRA weights
        peft_state_dict = get_peft_model_state_dict(transformer)
        
        sd3_lora_state_dict = {}
        for k, v in peft_state_dict.items():
            clean_key = k.replace('base_model.model.', '')
            sd3_lora_state_dict[f'transformer.{clean_key}'] = v

        save_file(sd3_lora_state_dict, os.path.join(output_dir, self._lora_weight_name))

        # Save learned embeddings (only for active text encoders)
        learned_embeds_dict = {}

        for idx, (text_encoder, ph_id) in enumerate(zip(text_encoders, placeholder_token_ids), start=1):
            if text_encoder is not None and ph_id is not None and idx in ti_text_encoders:
                embed_tensor = text_encoder.get_input_embeddings().weight[ph_id].detach().cpu()
                learned_embeds_dict[f'text_encoder_{idx}'] = embed_tensor

        save_file(learned_embeds_dict, os.path.join(output_dir, self._embeds_weight_name))

        print('\n--- Fine-Tuning Ended! ---')

        del transformer, pipe
        gc.collect()
        torch.cuda.empty_cache()

    def generate_personalized_image(
        self,
        lora_dir: str,
        prompt: str,
        placeholder_token: str = 'sks',
        initializer_token: str = 'dog',
        output_filename: str = 'result.png',
        device: str = 'cuda'
    ):
        print('--- Personalized Image Generation Started ---\n')

        if self._inference_pipe is None:
            print('[I 1/4] Loading base model for inference...')
            self._inference_pipe = StableDiffusion3Pipeline.from_pretrained(
                self._model_id,
                torch_dtype=torch.bfloat16
            ).to(device)

            # pipe.enable_model_cpu_offload()  # Automatic offloading to GPU (to avoid using only the GPU)
            # pipe.vae.enable_tiling()  # Tiling for VAE decoding (to save VRAM)

            print(f'[I 2/4] Loading learned embeddings and LoRA weights from \'{lora_dir}\' ...')

            # Load Textual Inversion Embeddings
            embeds_path = os.path.join(lora_dir, self._embeds_weight_name)

            if os.path.exists(embeds_path):
                learned_embeds = load_file(embeds_path)
                tokenizers = [self._inference_pipe.tokenizer, self._inference_pipe.tokenizer_2, self._inference_pipe.tokenizer_3]
                text_encoders = [self._inference_pipe.text_encoder, self._inference_pipe.text_encoder_2, self._inference_pipe.text_encoder_3]

                for idx, (tokenizer, text_encoder) in enumerate(zip(tokenizers, text_encoders), start=1):
                    if tokenizer is None or text_encoder is None:
                        continue

                    num_added_tokens = tokenizer.add_tokens(placeholder_token)
                    assert num_added_tokens == 1, (f'Placeholder token \'{placeholder_token}\' was not added as exactly one token.')

                    ph_id = tokenizer.convert_tokens_to_ids(placeholder_token)
                    text_encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)

                    key = f'text_encoder_{idx}'
                    if key in learned_embeds:
                        # If the encoder was trained, load the learned weights
                        text_encoder.get_input_embeddings().weight.data[ph_id] = learned_embeds[key].to(device, dtype=torch.bfloat16)
                    else:
                        # If the encoder was not trained, assign the initializer_token embedding
                        initializer_id = tokenizer.convert_tokens_to_ids(initializer_token)
                        text_encoder.get_input_embeddings().weight.data[ph_id] = text_encoder.get_input_embeddings().weight.data[initializer_id].clone()

            # Load LoRA weights
            self._inference_pipe.load_lora_weights(lora_dir, weight_name=self._lora_weight_name)

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