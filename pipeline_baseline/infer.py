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

    output_file = PipelineConfig.results_dir + str(len(os.listdir(PipelineConfig.results_dir))) + '.png'

    pipe = BaselinePipe()
    pipe.generate_personalized_image(
        lora_dir=PipelineConfig.adaptation_dir,
        prompt=PipelineConfig.generation_prompt,
        output_filename=output_file
    )