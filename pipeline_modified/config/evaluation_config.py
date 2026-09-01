class EvaluationConfig:
    """
    Set some configuration variables for the model evaluation process.
    """

    data_dir = 'data/'  # Relative path from project folder
    generation_dir = 'images_evaluation/modified'  # Generated images used during the evaluation process

    # Textual Inversion configuration: [1, 2] for two text encoders, [1, 2, 3] for three text encoders
    ti_text_encoders = [1, 2, 3]

    subject_cfgs = [
        { 'placeholder_token': '<sks>', 'class_token': 'backpack', 'living': False, 'training_steps': 1600 },
        { 'placeholder_token': '<sks>', 'class_token': 'backpack', 'living': False, 'training_steps': 1100 },
        { 'placeholder_token': '<sks>', 'class_token': 'stuffed animal', 'living': False, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'bowl', 'living': False, 'training_steps': 1000 },
        { 'placeholder_token': '<sks>', 'class_token': 'can', 'living': False, 'training_steps': 1400 },
        { 'placeholder_token': '<sks>', 'class_token': 'candle', 'living': False, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'cat', 'living': True, 'training_steps': 800 },
        { 'placeholder_token': '<sks>', 'class_token': 'cat', 'living': True, 'training_steps': 800 },
        { 'placeholder_token': '<sks>', 'class_token': 'clock', 'living': False, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'sneaker', 'living': False, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'dog', 'living': True, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'dog', 'living': True, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'dog', 'living': True, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'dog', 'living': True, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'dog', 'living': True, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'dog', 'living': True, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'dog', 'living': True, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'toy', 'living': False, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'boot', 'living': False, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'stuffed animal', 'living': False, 'training_steps': 1100 },
        { 'placeholder_token': '<sks>', 'class_token': 'toy', 'living': False, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'sunglasses', 'living': False, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'toy', 'living': False, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'toy', 'living': False, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'cartoon', 'living': True, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'toy', 'living': False, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'sneaker', 'living': False, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'teapot', 'living': False, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'vase', 'living': False, 'training_steps': 1200 },
        { 'placeholder_token': '<sks>', 'class_token': 'stuffed animal', 'living': False, 'training_steps': 1100 }
    ]

    training_prompts = [
        f'A photo of {cfg["placeholder_token"]} {cfg["class_token"]}'
        for cfg in subject_cfgs
    ]

    generation_prompts_live = [  # Keep the number of prompts the same as for 'generation_prompts_object'
        'A high quality studio photograph of {0} sitting on a sofa',
        'A high quality studio photograph of {0} running on a beach',
        'A high quality studio photograph of {0} wearing a small sombrero hat'
    ]

    generation_prompts_object = [  # Keep the number of prompts the same as for 'generation_prompts_live'
        'A high quality studio photograph of {0} on the surface of a frozen lake',
        'A high quality studio photograph of {0} on the grass with a yellow house in the background',
        'A high quality studio photograph of {0} on top of a dirt road'
    ]

    samples_per_prompt = 2