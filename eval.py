import os
import gc

import torch

from PIL import Image

from train import fine_tuning_lora
from infer import generate_personalized_image
from utils.CLIP_evaluator import CLIPEvaluator

from config.pipeline_config import PipelineConfig
from config.evaluation_config import EvaluationConfig


def test_subject_driven_pipeline(
    subject_idx,
    subject_name,
    token_identifier,
    data_dir,
    training_prompt,
    test_prompts,
    samples_per_prompt,
    adaptation_dir,
    generation_dir,
    evaluator
):

    '''
    Testing the model for each subject in the dataset:
    1. Fine-Tuning the model on the subject
    2. Generating N images on M prompt
    3. Computing average CLIP-I and CLIP-T on the generated images
    4. Asserting the resulting CLIP values
    '''

    print(f'\n--- Testing on subject \'{subject_name}\' ---\n')

    assert os.path.exists(data_dir), f'[ERROR] Data folder ({data_dir}) not found'

    print(f'[E 1/5] Loading reference images (from dataset)...')
    reference_images = []
    for file in os.listdir(data_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            img_path = os.path.join(data_dir, file)
            reference_images.append(Image.open(img_path).convert('RGB'))
            
    assert len(reference_images) > 0, f'({data_dir}) No images found'

    print(f'[E 2/5] Fine-Tuning...\n')
    fine_tuning_lora(
        image_folder=data_dir,
        output_dir=adaptation_dir,
        instance_prompt=training_prompt,
        max_train_steps=400,
        learning_rate=1e-4
    )
    
    weights_path = os.path.join(adaptation_dir, 'pytorch_lora_weights.safetensors')
    assert os.path.exists(weights_path), f'({weights_path}) No adaptation weights found'

    # Freeing up GPU memory before inference
    gc.collect()
    torch.cuda.empty_cache()

    print(f'\n[E 3/5] Generating images...\n')
    generation_dir = os.path.join(generation_dir, subject_name)
    os.makedirs(generation_dir, exist_ok=True)

    generated_samples = []

    for p_idx, prompt in enumerate(test_prompts):
        for s_idx in range(samples_per_prompt):
            output_filename = os.path.join(generation_dir, f'sample_P{p_idx}_S{s_idx}.png')
            
            generate_personalized_image(
                lora_dir=adaptation_dir,
                prompt=prompt.format(token_identifier),
                output_filename=output_filename
            )
            
            assert os.path.exists(output_filename), f'({output_filename}) Generated image not found'
            gen_img = Image.open(output_filename).convert('RGB')
            generated_samples.append((gen_img, prompt))

            gc.collect()
            torch.cuda.empty_cache()

    print(f'[OK] {len(generated_samples)} samples generated for \'{subject_name}\'')

    print(f'\n[E 4/5] Computing CLIP metrics (on all the samples)...')
    clip_t_scores = []
    clip_i_scores = []

    for gen_img, prompt in generated_samples:
        score_t = evaluator.compute_clip_t(gen_img, prompt)
        clip_t_scores.append(score_t)

        score_i = evaluator.compute_clip_i(gen_img, reference_images)
        clip_i_scores.append(score_i)

    mean_clip_t = sum(clip_t_scores) / len(clip_t_scores)
    mean_clip_i = sum(clip_i_scores) / len(clip_i_scores)

    print(f'\n--------------------------------------------------')
    print(f' RESULTS FOR \'{subject_name}\':')
    print(f' - Total generated samples : {len(generated_samples)}')
    print(f' - Avg CLIP-T (Text)       : {mean_clip_t:.4f}')
    print(f' - Avg CLIP-I (Image)      : {mean_clip_i:.4f}')
    print(f'--------------------------------------------------')

    if mean_clip_t < 0.25:
        f'[WARN] Average CLIP-T ({mean_clip_t:.4f}) not sufficient for \'{subject_idx}\' (thresh: 0.25)'
    
    if mean_clip_i < 0.70:
        f'[WARN] Average CLIP-I ({mean_clip_i:.4f}) not sufficient for \'{subject_idx}\' (thresh: 0.70)'


if __name__ == '__main__':
    for subject_idx, subject_name in enumerate(os.listdir(EvaluationConfig.data_dir)):
        data_dir = os.path.join(EvaluationConfig.data_dir, subject_name)

        if EvaluationConfig.subject_cfgs[subject_idx]['living']:
            test_prompts = EvaluationConfig.generation_prompts_live
        else:
            test_prompts = EvaluationConfig.generation_prompts_object

        test_subject_driven_pipeline(
            subject_idx=subject_idx,
            subject_name=subject_name,
            token_identifier=EvaluationConfig.subject_cfgs[subject_idx]['token_identifier'],
            data_dir=data_dir,
            training_prompt=EvaluationConfig.training_prompts[subject_idx],
            test_prompts=test_prompts,
            samples_per_prompt=EvaluationConfig.samples_per_prompt,
            adaptation_dir=PipelineConfig.adaptation_dir,
            generation_dir=EvaluationConfig.generation_dir,
            evaluator=CLIPEvaluator()
        )
