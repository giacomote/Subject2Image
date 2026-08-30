import os

import sys
from pathlib import Path

# Loading local files
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipeline_modified.pipeline import ModifiedPipe
from pipeline_modified.config.pipeline_config import PipelineConfig


if __name__ == '__main__':
    os.makedirs(PipelineConfig.results_dir, exist_ok=True)

    output_file = PipelineConfig.results_dir + str(len(os.listdir(PipelineConfig.results_dir))) + '.png'

    pipe = ModifiedPipe()
    pipe.generate_personalized_image(
        lora_dir=PipelineConfig.adaptation_dir,
        prompt=PipelineConfig.generation_prompt,
        placeholder_token=PipelineConfig.placeholder_token,
        initializer_token=PipelineConfig.class_token,
        output_filename=output_file
    )