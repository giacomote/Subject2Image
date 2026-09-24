# Subject2Image

A Computer Vision project focused on subject-driven image generation.

The goal of this project is to **explore subject-driven image generation** through a series of experiments, implementing
two pipelines capable of generating diverse images featuring a specific, user-defined subject while preserving high
visual consistency across different contexts and scenarios.

## Overview

This project implements **3 distinct pipelines** for subject-driven image generation:
- **Generation Only**: An unpersonalized pipeline (useful to compare the personalization results)
- **LoRA Baseline**: The reference LoRA implementation that establishes the foundational approach to subject-driven 
  generation, providing a solid baseline for comparison and evaluation.
- **LoRA Custom**: An experimental variant built upon the *LoRA Baseline* pipeline, incorporating a data augmentation
  stage to increase the number of subject images used during training.
- **TI Custom**: An experimental Textual Inversion pipeline, based on Stable Diffusion 3.5.

All the pipelines (except *Generation Only*) follow the same workflow (training, inference, and evaluation) and can be
used independently.

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

    Subject2Image/
    ├── 00_docs/              # Documentation (paper / slideshow / qualitative comparison)
    ├── data/                 # Datasets (create this directory before use)
    ├── hpc/                  # Scripts for running experiments on HPC clusters
    ├── metrics/              # Evaluation metrics
    ├── pipelines/
    │   ├── gen_only/
    │   ├── lora_baseline/
    │   ├── lora_custom/
    │   └── ti_custom/
    ├── .gitignore
    ├── LICENSE
    ├── README.md
    └── requirements.txt      # Python requirements

> [!NOTE]
> Some more folders are automatically created while running project scripts.  
> Those folders are needed to save results and temporary files.  
> All of them are specified in the pipelines configurations files.

> [!NOTE]
> Pipeline configuration files are located inside the relative pipeline `config/` folder.  
> Please check them before running any experiment.

## 🛠️ Installation

### 1. Clone the Repository

```bash
>> git clone https://github.com/giacomote/Subject2Image.git
>> cd Subject2Image
```

### 2. Create a Virtual Environment

It is recommended to use a dedicated Python virtual environment to avoid dependency conflicts.

```bash
>> python -m venv .venv
>> source .venv/bin/activate  # .\.venv\Scripts\activate (on Windows)
```

### 3. Install Dependencies

Install the required Python packages using the provided `requirements.txt` file.

```bash
>> pip install -r requirements.txt
```

### 4. Prepare the Project Structure

Before running the pipelines, create the required directories if they do not already exist.  
In this case, only the `data/` directory is required before running the project.

```bash
>> mkdir -p data
```

## 🚀 Usage

### 1. Dataset Preparation

#### Training & Inference Experiments

For **training and inference**, place **4-6 images** of your target subject directly in the `data/` directory.  
For optimal results, use images featuring varied angles, lighting conditions, and backgrounds.

#### Evaluation Experiments

For **evaluation**, organize the dataset into **separate subdirectories**, with one subdirectory per subject.  
Each subject should contain 4-6 images of that subject.

For example:

```text
data/
├── subject_01/
│   ├── 01.png
│   ├── 02.png
│   └── ...
└── subject_02/
    ├── 01.png
    ├── 02.png
    └── ...
```

The evaluation script automatically iterates over the subject subdirectories, trains the model separately for each
subject, generates the corresponding images, and computes the evaluation metrics.

### 2. Check Configuration

Before running training or inference or launch an evaluation experiment, review the configuration files located in the
respective pipeline `config/` directories.

If you intend to launch the experiment on a remote HPC cluster, please check the launch scripts in the `hpc/` directory.

### 3a. Training & Inference

The training script fine-tunes the model on a specific subject.   
Once the model has been successfully trained, the inference script can be used to generate new images featuring the
same subject.

The training step only needs to be performed once for each subject.  
Afterward, the trained model can be reused to generate as many image variations as needed.

> [!NOTE]
> As it is unpersonalized, *Generation Only* pipeline **does not have a training script**.

#### Local Execution

```bash
>> cd <clone_dir>/Subject2Image  # Place yourself at the root of the repository
>> python pipelines/<specific_pipeline_dir>/train.py
>> python pipelines/<specific_pipeline_dir>/infer.py
```

#### HPC Cluster Execution

```bash
>> cd <clone_dir>/Subject2Image  # Place yourself at the root of the repository
>> sbatch hpc/run_training.sh <pipeline_dir_name>
>> sbatch hpc/run_inference.sh <pipeline_dir_name>
```

### 3b. Evaluation

The evaluation workflow is designed to automatically assess the performance of the pipeline across multiple subjects.

The evaluation script iterates over the different subject folders contained in the `data/` directory.  
For each subject, it automatically trains the model, generates a set of images, and computes the evaluation metrics
used to measure the quality and consistency of the generated results.

This workflow allows the performance of the pipeline to be evaluated systematically across multiple subjects without
requiring manual training and inference for each one.

#### Local Execution

```bash
>> cd <clone_dir>/Subject2Image  # Place yourself at the root of the repository
>> python pipelines/<specific_pipeline_dir>/eval.py
```

#### HPC Cluster Execution

```bash
>> cd <clone_dir>/Subject2Image  # Place yourself at the root of the repository
>> sbatch hpc/run_evaluation.sh <pipeline_dir_name>
```

## 📈 Results

The generated images and evaluation results are automatically organized into dedicated output directories.

> [!NOTE]
> All the cited directories are the default ones.  
> If you change them in the pipelines configuration files they may be different.

### Training & Inference

During the **training and inference workflow**, the generated images are saved in the `images_inference/` directory.  
This folder contains the different image variations produced by the model for the selected subject.

### Evaluation

During the **evaluation workflow**, the generated images for each subject are saved in the `images_evaluation/`
directory.

The evaluation metrics computed for each subject and experiment are reported in the evaluation output.  
When running the evaluation on an **HPC cluster**, the metrics are available in the corresponding log files stored in
the `logs/` directory. When running the evaluation **locally**, instead, the metrics are instead displayed directly in
the script's console output.