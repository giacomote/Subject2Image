import os
import gc

import sys
from pathlib import Path

import torch

from PIL import Image

# Filtering Warnings
import warnings

warnings.filterwarnings(category=UserWarning, action='ignore')
warnings.filterwarnings(category=FutureWarning, action='ignore')

# Loading local files
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipeline_baseline.pipeline import BaselinePipe
from metrics.subject_metrics import SubjectMetrics
from metrics.dataset_metrics import DatasetMetrics

from pipeline_baseline.config.evaluation_config import EvaluationConfig


def train_and_generate_for_subject(
    subject_name: str,
    token_identifier: str,
    data_dir: str,
    training_prompt: str,
    test_prompts: list[str],
    samples_per_prompt: int,
    adaptation_dir: str,
    subject_gen_dir: str
) -> str:

    print(f'\n--- Training and generating on subject \'{subject_name}\' ---\n')

    assert os.path.exists(data_dir), f'[ERROR] Data folder ({data_dir}) not found'

    print(f'[TG 1/2] Fine-Tuning (LoRA)...\n')
    pipe = BaselinePipe()

    pipe.fine_tuning_lora(
        image_folder=data_dir,
        output_dir=adaptation_dir,
        instance_prompt=training_prompt,
        max_train_steps=1200,
        learning_rate=1e-4
    )
    
    weights_path = os.path.join(adaptation_dir, 'base_lora_weights.safetensors')
    assert os.path.exists(weights_path), f'({weights_path}) No adaptation weights found'

    # Freeing up GPU memory before inference
    gc.collect()
    torch.cuda.empty_cache()

    print(f'\n[TG 2/2] Generating images...\n')
    subject_gen_dir = os.path.join(subject_gen_dir, subject_name)
    os.makedirs(subject_gen_dir, exist_ok=True)

    generated_count = 0
    for p_idx, prompt in enumerate(test_prompts):
        for s_idx in range(samples_per_prompt):
            output_filename = os.path.join(subject_gen_dir, f'sample_P{p_idx}_S{s_idx}.png')
            
            pipe.generate_personalized_image(
                lora_dir=adaptation_dir,
                prompt=prompt.format(token_identifier),
                output_filename=output_filename
            )
            
            assert os.path.exists(output_filename), f'({output_filename}) Generated image not found'
            generated_count += 1

            gc.collect()
            torch.cuda.empty_cache()

    print(f'[OK] {generated_count} samples generated for \'{subject_name}\'')
    return subject_gen_dir


def evaluate_subject_metrics(
    subject_name: str,
    class_token: str,
    data_dir: str,
    subject_gen_dir: str,
    test_prompts: list[str],
    samples_per_prompt: int,
    evaluator: SubjectMetrics
) -> tuple[float, float, float, float]:

    print(f'\n--- Computing subject metrics on subject \'{subject_name}\' ---\n')

    print('[SM 1/2] Loading reference images...')
    reference_images = []
    for file in os.listdir(data_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            reference_images.append(Image.open(os.path.join(data_dir, file)).convert('RGB'))

    assert len(reference_images) > 0, f'({data_dir}) No reference images found'

    print('[SM 2/2] Computing CLIP, DINO, LPIPS metrics...')
    clip_t_scores = []
    clip_i_scores = []
    dino_i_scores = []
    lpips_scores = []

    for p_idx, prompt in enumerate(test_prompts):
        formatted_prompt = prompt.format(class_token)

        for s_idx in range(samples_per_prompt):
            sample_path = os.path.join(subject_gen_dir, f'sample_P{p_idx}_S{s_idx}.png')
            gen_img = Image.open(sample_path).convert('RGB')

            clip_t_scores.append(evaluator.compute_clip_t(gen_img, formatted_prompt))
            clip_i_scores.append(evaluator.compute_clip_i(gen_img, reference_images))
            dino_i_scores.append(evaluator.compute_dino_i(gen_img, reference_images))
            lpips_scores.append(evaluator.compute_lpips(gen_img, reference_images))

    mean_clip_t = sum(clip_t_scores) / len(clip_t_scores)
    mean_clip_i = sum(clip_i_scores) / len(clip_i_scores)
    mean_dino_i = sum(dino_i_scores) / len(dino_i_scores)
    mean_lpips = sum(lpips_scores) / len(lpips_scores)

    print('--------------------------------------------------')
    print(f' RESULTS FOR \'{subject_name}\':')
    print(f' - Avg CLIP-T (Text)  : {mean_clip_t:.4f}')
    print(f' - Avg CLIP-I (Image) : {mean_clip_i:.4f}')
    print(f' - Avg DINO-I (Image) : {mean_dino_i:.4f}')
    print(f' - Avg LPIPS (Image)  : {mean_lpips:.4f}')
    print('--------------------------------------------------')


    return mean_clip_t, mean_clip_i, mean_dino_i, mean_lpips


def evaluate_dataset_metrics(
    real_dataset_dir: str,
    gen_dataset_dir: str,
    evaluator: DatasetMetrics
) -> tuple[float, float]:

    print('\n--- Computing Dataset-Level Metrics (FID & KID) ---\n')

    print('[DM 1/2] Loading dataset images and extracting Inception features...')
    real_feats, gen_feats = evaluator.extract_dataset_features(real_dataset_dir, gen_dataset_dir)

    print('\n[DM 2/2] Computing Dataset FID and KID...')
    fid_score = evaluator.compute_fid(real_feats, gen_feats)
    kid_score = evaluator.compute_kid(real_feats, gen_feats)

    return fid_score, kid_score


if __name__ == '__main__':
    subject_evaluator = SubjectMetrics()
    
    dataset_clip_t_scores = []
    dataset_clip_i_scores = []
    dataset_dino_i_scores = []
    dataset_lpips_scores = []
    subject_count = 0

    subject_folders = sorted([
        f for f in os.listdir(EvaluationConfig.data_dir)
        if os.path.isdir(os.path.join(EvaluationConfig.data_dir, f))
    ])

    for subject_idx, subject_name in enumerate(subject_folders):
        data_dir = os.path.join(EvaluationConfig.data_dir, subject_name)

        if EvaluationConfig.subject_cfgs[subject_idx]['living']:
            test_prompts = EvaluationConfig.generation_prompts_live
        else:
            test_prompts = EvaluationConfig.generation_prompts_object

        placeholder_token = EvaluationConfig.placeholder_token
        class_token = EvaluationConfig.subject_cfgs[subject_idx]["class_token"]
        token_identifier = f'{placeholder_token} {class_token}'

        # Training the model and generating images for each subject
        subject_gen_dir = train_and_generate_for_subject(
            subject_name=subject_name,
            token_identifier=token_identifier,
            data_dir=data_dir,
            training_prompt=EvaluationConfig.training_prompts[subject_idx],
            test_prompts=test_prompts,
            samples_per_prompt=EvaluationConfig.samples_per_prompt,
            adaptation_dir=EvaluationConfig.adaptation_dir,
            subject_gen_dir=EvaluationConfig.generation_dir
        )

        # Computing subject metrics
        sub_clip_t, sub_clip_i, sub_dino_i, sub_lpips = evaluate_subject_metrics(
            subject_name=subject_name,
            class_token=class_token,
            data_dir=data_dir,
            subject_gen_dir=subject_gen_dir,
            test_prompts=test_prompts,
            samples_per_prompt=EvaluationConfig.samples_per_prompt,
            evaluator=subject_evaluator
        )

        dataset_clip_t_scores.append(sub_clip_t)
        dataset_clip_i_scores.append(sub_clip_i)
        dataset_dino_i_scores.append(sub_dino_i)
        dataset_lpips_scores.append(sub_lpips)
        subject_count += 1

        gc.collect()
        torch.cuda.empty_cache()

    if subject_count > 0:
        dataset_mean_clip_t = sum(dataset_clip_t_scores) / subject_count
        dataset_mean_clip_i = sum(dataset_clip_i_scores) / subject_count
        dataset_mean_dino_i = sum(dataset_dino_i_scores) / subject_count
        dataset_mean_lpips = sum(dataset_lpips_scores) / subject_count
        
        # Freeing memory from CLIP and DINO evaluators
        del subject_evaluator
        gc.collect()
        torch.cuda.empty_cache()

        # Computing Dataset-Level Metrics (FID & KID)
        dataset_evaluator = DatasetMetrics()
        global_fid, global_kid = evaluate_dataset_metrics(
            real_dataset_dir=EvaluationConfig.data_dir,
            gen_dataset_dir=EvaluationConfig.generation_dir,
            evaluator=dataset_evaluator
        )

        print('\n**************************************************')
        print(' FINAL DATASET EVALUATION RESULTS:')
        print(f' - Total tested subjects      : {subject_count}')
        print(f' - Dataset Avg CLIP-T (Text)  : {dataset_mean_clip_t:.4f}')
        print(f' - Dataset Avg CLIP-I (Image) : {dataset_mean_clip_i:.4f}')
        print(f' - Dataset Avg DINO-I (Image) : {dataset_mean_dino_i:.4f}')
        print(f' - Dataset Avg LPIPS (Image)  : {dataset_mean_lpips:.4f}')
        print(f' - Global Dataset FID         : {global_fid:.4f}')
        print(f' - Global Dataset KID         : {global_kid:.4f}')
        print('**************************************************')
    else:
        print('\n[WARN] No subjects found in the dataset folder\n')