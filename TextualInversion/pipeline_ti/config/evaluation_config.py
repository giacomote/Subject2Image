class EvaluationConfig:
    """
    Set some configuration variables for the model evaluation process.
    """

    # Folder names must end with a '/' character
    # They can be either absolute paths or relative paths (starting from the repository folder)
    data_dir = 'data/'
    adaptation_dir = 'adaptation_evaluation/'
    generation_dir = 'images_evaluation/baseline_ti/'  # Generated images used during the evaluation process

    placeholder_token = '<sks>'

    subject_cfgs = [
        { 'class_token': 'backpack', 'living': False },
        { 'class_token': 'backpack', 'living': False },
        { 'class_token': 'stuffed animal', 'living': False },
        { 'class_token': 'bowl', 'living': False },
        { 'class_token': 'can', 'living': False },
        { 'class_token': 'candle', 'living': False },
        { 'class_token': 'cat', 'living': True },
        { 'class_token': 'cat', 'living': True },
        { 'class_token': 'clock', 'living': False },
        { 'class_token': 'sneaker', 'living': False },
        { 'class_token': 'dog', 'living': True },
        { 'class_token': 'dog', 'living': True },
        { 'class_token': 'dog', 'living': True },
        { 'class_token': 'dog', 'living': True },
        { 'class_token': 'dog', 'living': True },
        { 'class_token': 'dog', 'living': True },
        { 'class_token': 'dog', 'living': True },
        { 'class_token': 'toy', 'living': False },
        { 'class_token': 'boot', 'living': False },
        { 'class_token': 'stuffed animal', 'living': False },
        { 'class_token': 'toy', 'living': False },
        { 'class_token': 'sunglasses', 'living': False },
        { 'class_token': 'toy', 'living': False },
        { 'class_token': 'toy', 'living': False },
        { 'class_token': 'cartoon', 'living': True },
        { 'class_token': 'toy', 'living': False },
        { 'class_token': 'sneaker', 'living': False },
        { 'class_token': 'teapot', 'living': False },
        { 'class_token': 'vase', 'living': False },
        { 'class_token': 'stuffed animal', 'living': False }
    ]

    training_prompts = []
    for cfg in subject_cfgs:
        training_prompts.append(f'A photo of {placeholder_token} {cfg["class_token"]}')

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