# NeuralVisionaries

A Computer Vision project focused on subject-driven image generation.

The goal of this project is to build a pipeline capable of generating diverse images featuring a specific, user-defined
subject while maintaining high visual consistency.

## 📜 Author & License

This repository and its files are provided by Giacomo Tessari (`@giacomote` on GitHub) and Sofia Caruso (`@SofiaC27` on
GitHub), and they are released under the "GPL v3" license.

    Copyright (C) 2026 Giacomo Tessari, Sofia Caruso

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.

For more details about the license and to view the full terms and conditions, please refer to the [LICENSE](LICENSE)
file.

## 📂 Project Structure

    NeuralVisionaries/
    ├── adaptation/           # Fine-tuned model weights - once trained the model
    ├── config/               # Configuration variables
    ├── data/                 # Default datasets folder - to create before use
    ├── evaluation_images/    # Temporary images generated during the evaluation process
    ├── hpc/                  # HPC cluster access scripts
    ├── logs/                 # Logs (output and errors) - once ran training/inference
    ├── results/              # Generated images - once ran the inference process
    ├── utils/                # Various scripts and models, used around the project
    ├── .gitignore
    ├── eval.py               # Pipeline evaluation script
    ├── infer.py              # Inference script
    ├── LICENSE
    ├── README.md
    ├── requirements.txt      # Python requirements
    └── train.py              # Training script

## 🚀 How to Run the Project

To run the project, follow these simple steps to set up your data and execute the training and inference pipelines.

### 1. Dataset Preparation

Create a `data/` directory in the root of your project and populate it with **5 to 10 images** of your target subject.  
For optimal results, ensure the images feature varied angles, lighting, and backgrounds.

```bash
>> cd <...>/NeuralVisionaries
>> mkdir data/
```

### 2. Configuration

Make sure that the configuration variables in `config/pipeline_config.py` are properly set.

### 3. Training, Inference & Evaluation

Run the training script first to fine-tune the model on your subject, then use the inference script to generate new
images.

Alternatively, you can run the evaluation script to visualize the pipeline performances.
That script will train the model and generate multiple images, to compute then some specific metrics.

You can execute the scripts either locally or on an HPC cluster depending on your available compute resources.

> [!NOTE]
> Once the model has been successfully trained for a specific subject, the training step can be skipped entirely.  
> You can run the inference script as many times as you like to generate new variations.

#### 🖥️ Local Execution

```bash
>> cd <...>/NeuralVisionaries
>> python -m venv .venv
>> source .venv/bin/activate  # .\.venv\Scripts\activate (on Windows)
>> pip install -r requirements.txt

>> python train.py
>> python infer.py
>> python eval.py
```

#### ⚡ HPC Cluster Execution

```bash
>> cd <...>/NeuralVisionaries
>> python -m venv .venv
>> source .venv/bin/activate
>> pip install -r requirements.txt

>> sbatch hpc/run_training.sh
>> sbatch hpc/run_inference.sh
>> sbatch hpc/run_evaluation.sh
```