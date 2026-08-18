import os
import gc

import torch

from PIL import Image

from train import fine_tuning_lora
from infer import generate_personalized_image
from utils.subject_metrics import SubjectMetrics

from config.pipeline_config import PipelineConfig
from config.evaluation_config import EvaluationConfig


def test_subject_driven_pipeline(
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

    """
    Testing the model for each subject in the dataset:
    1. Fine-Tuning the model on the subject
    2. Generating N images on M prompt
    3. Computing average CLIP-I and CLIP-T on the generated images
    4. Asserting the resulting CLIP values
    """

    print(f'\n--- Testing on subject \'{subject_name}\' ---\n')

    assert os.path.exists(data_dir), f'[ERROR] Data folder ({data_dir}) not found'

    print(f'[E 1/4] Loading reference images (from dataset)...')
    reference_images = []
    for file in os.listdir(data_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            img_path = os.path.join(data_dir, file)
            reference_images.append(Image.open(img_path).convert('RGB'))
            
    assert len(reference_images) > 0, f'({data_dir}) No images found'

    print(f'[E 2/4] Fine-Tuning...\n')
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

    print(f'\n[E 3/4] Generating images...\n')
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

    print(f'\n[E 4/4] Computing CLIP & DINO metrics (on all the samples)...')
    clip_t_scores = []
    clip_i_scores = []
    dino_i_scores = []

    for gen_img, prompt in generated_samples:
        score_t = evaluator.compute_clip_t(gen_img, prompt)
        clip_t_scores.append(score_t)

        score_i = evaluator.compute_clip_i(gen_img, reference_images)
        clip_i_scores.append(score_i)

        score_dino = evaluator.compute_dino_i(gen_img, reference_images)
        dino_i_scores.append(score_dino)

    mean_clip_t = sum(clip_t_scores) / len(clip_t_scores)
    mean_clip_i = sum(clip_i_scores) / len(clip_i_scores)
    mean_dino_i = sum(dino_i_scores) / len(dino_i_scores)

    print('\n--------------------------------------------------')
    print(f' RESULTS FOR \'{subject_name}\':')
    print(f' - Total generated samples : {len(generated_samples)}')
    print(f' - Avg CLIP-T (Text)       : {mean_clip_t:.4f}')
    print(f' - Avg CLIP-I (Image)      : {mean_clip_i:.4f}')
    print(f' - Avg DINO-I (Image)      : {mean_dino_i:.4f}')
    print('--------------------------------------------------')

    if mean_clip_t < 0.25:
        print(f'[WARN] Average CLIP-T ({mean_clip_t:.4f}) not sufficient for \'{subject_name}\' (thresh: 0.25)')
    
    if mean_clip_i < 0.70:
        print(f'[WARN] Average CLIP-I ({mean_clip_i:.4f}) not sufficient for \'{subject_name}\' (thresh: 0.70)')

    if mean_dino_i < 0.60:
        print(f'[WARN] Average DINO-I ({mean_dino_i:.4f}) not sufficient for \'{subject_name}\' (thresh: 0.60)')

    return mean_clip_t, mean_clip_i, mean_dino_i


if __name__ == '__main__':
    global_evaluator = SubjectMetrics()
    
    dataset_clip_t_scores = []
    dataset_clip_i_scores = []
    dataset_dino_i_scores = []
    subject_count = 0

    for subject_idx, subject_name in enumerate(os.listdir(EvaluationConfig.data_dir)):
        data_dir = os.path.join(EvaluationConfig.data_dir, subject_name)

        if EvaluationConfig.subject_cfgs[subject_idx]['living']:
            test_prompts = EvaluationConfig.generation_prompts_live
        else:
            test_prompts = EvaluationConfig.generation_prompts_object

        sub_clip_t, sub_clip_i, sub_dino_i = test_subject_driven_pipeline(
            subject_name=subject_name,
            token_identifier=EvaluationConfig.subject_cfgs[subject_idx]['token_identifier'],
            data_dir=data_dir,
            training_prompt=EvaluationConfig.training_prompts[subject_idx],
            test_prompts=test_prompts,
            samples_per_prompt=EvaluationConfig.samples_per_prompt,
            adaptation_dir=PipelineConfig.adaptation_dir,
            generation_dir=EvaluationConfig.generation_dir,
            evaluator=global_evaluator
        )

        dataset_clip_t_scores.append(sub_clip_t)
        dataset_clip_i_scores.append(sub_clip_i)
        dataset_dino_i_scores.append(sub_dino_i)
        subject_count += 1

    if subject_count > 0:
        dataset_mean_clip_t = sum(dataset_clip_t_scores) / subject_count
        dataset_mean_clip_i = sum(dataset_clip_i_scores) / subject_count
        dataset_mean_dino_i = sum(dataset_dino_i_scores) / subject_count
        
        print('\n************************************************')
        print(' FINAL DATASET RESULTS:')
        print(f' - Total tested subjects      : {subject_count}')
        print(f' - Dataset Avg CLIP-T (Text)  : {dataset_mean_clip_t:.4f}')
        print(f' - Dataset Avg CLIP-I (Image) : {dataset_mean_clip_i:.4f}')
        print(f' - Dataset Avg DINO-I (Image) : {dataset_mean_dino_i:.4f}')
        print('**************************************************\n')
    else:
        print('\n[WARN] No subjects found in the dataset folder\n')