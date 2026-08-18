class EvaluationConfig:
    """
    Set some configuration variables for the model evaluation process.
    """

    data_dir = 'data/dreambooth/'  # Relative path from project folder
    generation_dir = 'evaluation_images/'  # Model will store the generated images used during the evaluation process

    subject_cfgs = [
        { 'token_identifier': 'sks backpack', 'living': False },
        { 'token_identifier': 'sks backpack', 'living': False },
        { 'token_identifier': 'sks stuffed animal', 'living': False },
        { 'token_identifier': 'sks bowl', 'living': False },
        { 'token_identifier': 'sks can', 'living': False },
        { 'token_identifier': 'sks candle', 'living': False },
        { 'token_identifier': 'sks cat', 'living': True },
        { 'token_identifier': 'sks cat', 'living': True },
        { 'token_identifier': 'sks clock', 'living': False },
        { 'token_identifier': 'sks sneaker', 'living': False },
        { 'token_identifier': 'sks dog', 'living': True },
        { 'token_identifier': 'sks dog', 'living': True },
        { 'token_identifier': 'sks dog', 'living': True },
        { 'token_identifier': 'sks dog', 'living': True },
        { 'token_identifier': 'sks dog', 'living': True },
        { 'token_identifier': 'sks dog', 'living': True },
        { 'token_identifier': 'sks dog', 'living': True },
        { 'token_identifier': 'sks toy', 'living': False },
        { 'token_identifier': 'sks boot', 'living': False },
        { 'token_identifier': 'sks stuffed animal', 'living': False },
        { 'token_identifier': 'sks toy', 'living': False },
        { 'token_identifier': 'sks sunglasses', 'living': False },
        { 'token_identifier': 'sks toy', 'living': False },
        { 'token_identifier': 'sks toy', 'living': False },
        { 'token_identifier': 'sks cartoon', 'living': True },
        { 'token_identifier': 'sks toy', 'living': False },
        { 'token_identifier': 'sks sneaker', 'living': False },
        { 'token_identifier': 'sks teapot', 'living': False },
        { 'token_identifier': 'sks vase', 'living': False },
        { 'token_identifier': 'sks stuffed animal', 'living': False }
    ]

    training_prompts = [
        f'A photo of {cfg["token_identifier"]}'
        for cfg in subject_cfgs
    ]

    generation_prompts_live = [  # Keep the number of prompts the same as for 'generation_prompts_object'
        'A high quality studio photograph of {0} sitting on a sofa',
        'A high quality studio photograph of {0} running on a beach',
        'A high quality studio photograph of {0} wearing a small top hat',
    ]

    generation_prompts_object = [  # Keep the number of prompts the same as for 'generation_prompts_live'
        'A high quality studio photograph of {0} on the surface of a frozen lake',
        'A high quality studio photograph of {0} on the grass with a yellow house in the background',
        'A high quality studio photograph of {0} on top of a dirt road',
    ]

    samples_per_prompt = 2