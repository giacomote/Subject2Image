import os

import sys
from pathlib import Path

# Loading local files
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipeline_baseline.pipeline import BaselinePipe
from pipeline_baseline.config.pipeline_config import PipelineConfig


if __name__ == '__main__':
    os.makedirs(PipelineConfig.results_dir, exist_ok=True)

    subject_id = PipelineConfig.data_dir.split('/')[-1]
    output_file = PipelineConfig.results_dir + str(len(os.listdir(PipelineConfig.results_dir))) + '_' + subject_id + '.png'

    token_identifier = f'{PipelineConfig.placeholder_token} {PipelineConfig.class_token}'
    generation_prompt = PipelineConfig.generation_prompt.format(token_identifier)

    pipe = BaselinePipe()
    pipe.generate_personalized_image(
        lora_dir=PipelineConfig.adaptation_dir,
        prompt=generation_prompt,
        output_filename=output_file
    )