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
    pipe = ModifiedPipe()

    if os.path.exists(PipelineConfig.data_dir):
        pipe.fine_tuning_lora(
            image_folder=PipelineConfig.data_dir,
            output_dir=PipelineConfig.adaptation_dir,
            instance_prompt=PipelineConfig.training_prompt,
            max_train_steps=PipelineConfig.training_steps,
            learning_rate=1e-4
        )
    else:
        print(f'[ERROR] Please create folder \'{PipelineConfig.data_dir}\' and insert 5-10 images of the subject')
