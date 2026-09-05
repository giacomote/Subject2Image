import os
import gc

import torch
import torch.nn.functional as F
import bitsandbytes as bnb

from torch.utils.data import DataLoader

from diffusers import StableDiffusion3Pipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from safetensors.torch import save_file, load_file

from pipeline_ti.custom_dataset import CustomDataset


class BaseTIPipe:
    def __init__(self):
        self._model_id = 'stabilityai/stable-diffusion-3.5-large'
        self._embeds_weight_name = 'ti_learned_embeds.safetensors'
        self._inference_pipe = None

    def _free_inference_memory(self):
        if self._inference_pipe is not None:
            del self._inference_pipe
            self._inference_pipe = None
            gc.collect()
            torch.cuda.empty_cache()

    def learn_embedding_ti(
        self,
        image_folder: str,
        output_dir: str,
        placeholder_token: str = '<sks>',
        initializer_token: str = 'dog',
        instance_prompt: str = 'A photo of <sks> dog',
        max_train_steps: int = 1200,
        ti_learning_rate: float = 1e-4,
        device: str = 'cuda'
    ):
        self._free_inference_memory()

        print('--- Textual Inversion Started ---\n')
        os.makedirs(output_dir, exist_ok=True)

        print('[T 1/8] Loading base model...')
        pipe = StableDiffusion3Pipeline.from_pretrained(
            self._model_id,
            dtype=torch.bfloat16
        ).to(device)

        print(f'[T 2/8] Registering placeholder token \'{placeholder_token}\' in tokenizers and text encoders...')
        tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
        text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
        placeholder_token_ids = []

        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            if tokenizer is None or text_encoder is None:
                placeholder_token_ids.append(None)
                continue

            num_added_tokens = tokenizer.add_tokens(placeholder_token)
            assert num_added_tokens == 1, (f'[ERROR] Placeholder token \'{placeholder_token}\' was not added as exactly one token')

            placeholder_id = tokenizer.convert_tokens_to_ids(placeholder_token)
            placeholder_token_ids.append(placeholder_id)

            # Resize embedding table
            text_encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)

            # Copy initializer token weights
            initializer_id = tokenizer.convert_tokens_to_ids(initializer_token)
            token_embeds = text_encoder.get_input_embeddings().weight.data
            token_embeds[placeholder_id] = token_embeds[initializer_id]

            # Freeze text encoder except input embeddings
            text_encoder.requires_grad_(False)
            text_encoder.get_input_embeddings().requires_grad_(True)

        if pipe.text_encoder_3 is not None:
            pipe.text_encoder_3.requires_grad_(False)
            pipe.text_encoder_3.eval()

        print('[T 3/8] Encoding images (in latent space using VAE)...')
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

        print('[T 4/8] Unloading VAE from GPU to save VRAM...')
        del pipe.vae
        gc.collect()
        torch.cuda.empty_cache()

        print('[T 5/8] Preparing Transformer and Parameters...')
        transformer = pipe.transformer.to(device)

        transformer.requires_grad_(False)
        transformer.eval()

        transformer.enable_gradient_checkpointing()

        # Collect input embeddings parameters from text encoders
        ti_params = []
        for text_encoder in text_encoders:
            if text_encoder is not None:
                ti_params.extend(list(text_encoder.get_input_embeddings().parameters()))

        print('[T 6/8] Applying AdamW 8-bit optimization...')
        try:
            optimizer = bnb.optim.AdamW8bit(ti_params, lr=ti_learning_rate)
        except Exception:
            optimizer = torch.optim.AdamW(ti_params, lr=ti_learning_rate)
            print('[WARN] Cannot apply AdamW 8-bit: falling back to standard AdamW')

        # LR Scheduler
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(max_train_steps * 0.10),
            num_training_steps=max_train_steps
        )

        print(f'[T 7/8] Training loop ({max_train_steps} steps)...')
        global_step = 0
        num_samples = len(cached_latents)

        prompt_t5 = instance_prompt.replace(placeholder_token, "")
        prompt_t5 = " ".join(prompt_t5.split())

        while global_step < max_train_steps:
            optimizer.zero_grad()

            prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=instance_prompt,
                prompt_2=instance_prompt,
                prompt_3=prompt_t5,
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

            # Zero out gradients for all tokens except the placeholder token across text encoders
            for text_encoder, ph_id in zip(text_encoders, placeholder_token_ids):
                if text_encoder is not None and ph_id is not None:
                    grads = text_encoder.get_input_embeddings().weight.grad
                    if grads is not None:
                        index_grads_to_zero = torch.ones(grads.shape[0], dtype=torch.bool, device=grads.device)
                        index_grads_to_zero[ph_id] = False
                        grads[index_grads_to_zero] = 0

            optimizer.step()
            lr_scheduler.step()

            global_step += 1
            if global_step % 50 == 0 or global_step == max_train_steps:
                current_lr = lr_scheduler.get_last_lr()[0]
                print(f'\tStep [{global_step}/{max_train_steps}] - Loss: {loss.item():.4f} - TI LR: {current_lr:.6f}')

        print(f'\n[T 8/8] Saving learned embeddings to: \'{output_dir}\' ...')

        # Save learned embeddings for text encoders
        learned_embeds_dict = {}
        for idx, (text_encoder, ph_id) in enumerate(zip(text_encoders, placeholder_token_ids), start=1):
            if text_encoder is not None and ph_id is not None:
                embed_tensor = text_encoder.get_input_embeddings().weight[ph_id].detach().cpu()
                learned_embeds_dict[f'text_encoder_{idx}'] = embed_tensor

        save_file(learned_embeds_dict, os.path.join(output_dir, self._embeds_weight_name))

        print('\n--- Textual Inversion Ended! ---')

        del transformer, pipe
        gc.collect()
        torch.cuda.empty_cache()

    def generate_personalized_image_ti(
        self,
        embeds_dir: str,
        prompt: str,
        placeholder_token: str = '<sks>',
        initializer_token: str = 'dog',
        output_filename: str = 'result.png',
        device: str = 'cuda'
    ):
        print('--- Personalized Image Generation Started ---\n')

        if self._inference_pipe is None:
            print('[I 1/3] Loading base model for inference...')
            self._inference_pipe = StableDiffusion3Pipeline.from_pretrained(
                self._model_id,
                dtype=torch.bfloat16
            ).to(device)

            print(f'[I 2/3] Loading learned embeddings from \'{embeds_dir}\' ...')

            # Load Textual Inversion Embeddings
            embeds_path = os.path.join(embeds_dir, self._embeds_weight_name)

            if os.path.exists(embeds_path):
                learned_embeds = load_file(embeds_path)
                tokenizers = [self._inference_pipe.tokenizer, self._inference_pipe.tokenizer_2]
                text_encoders = [self._inference_pipe.text_encoder, self._inference_pipe.text_encoder_2]

                for idx, (tokenizer, text_encoder) in enumerate(zip(tokenizers, text_encoders), start=1):
                    if tokenizer is None or text_encoder is None:
                        continue

                    num_added_tokens = tokenizer.add_tokens(placeholder_token)
                    assert num_added_tokens == 1, (f'[ERROR] Placeholder token \'{placeholder_token}\' was not added as exactly one token')

                    ph_id = tokenizer.convert_tokens_to_ids(placeholder_token)
                    text_encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)

                    key = f'text_encoder_{idx}'
                    if key in learned_embeds:
                        text_encoder.get_input_embeddings().weight.data[ph_id] = learned_embeds[key].to(device, dtype=torch.bfloat16)
                    else:
                        initializer_id = tokenizer.convert_tokens_to_ids(initializer_token)
                        text_encoder.get_input_embeddings().weight.data[ph_id] = text_encoder.get_input_embeddings().weight.data[initializer_id].clone()
        else:
            print('[I 1/3] Reusing cached base model...')
            print('[I 2/3] Reusing cached adaptation weights...')

        print('[I 3/3] Generating image...')

        prompt_t5 = prompt.replace(placeholder_token, "")
        prompt_t5 = " ".join(prompt_t5.split())

        image = self._inference_pipe(
            prompt=prompt,
            prompt_2=prompt,      
            prompt_3=prompt_t5,
            negative_prompt='blurry, distorted, low quality, bad anatomy',
            num_inference_steps=28,
            guidance_scale=4.5,
            width=1024,
            height=1024
        ).images[0]

        image.save(output_filename)
        print(f'[OK] Image saved: {output_filename}')

        print('\n--- Personalized Image Generation Ended! ---')