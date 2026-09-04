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
    token_identifier = f'{PipelineConfig.placeholder_token} {PipelineConfig.class_token}'
    training_prompt = PipelineConfig.training_prompt.format(token_identifier)

    if os.path.exists(PipelineConfig.data_dir):
        pipe = ModifiedPipe()
        pipe.fine_tuning_lora(
            image_folder=PipelineConfig.data_dir,
            output_dir=PipelineConfig.adaptation_dir,
            instance_prompt=training_prompt
        )
    else:
        print(f'[ERROR] Please create folder \'{PipelineConfig.data_dir}\' and insert 5-10 images of the subject')
