class EvaluationConfig:
    """
    Set some configuration variables for the model evaluation process.
    """

    data_dir = 'data/dreambooth/'  # Relative path from project folder
    generation_dir = 'evaluation_images/'  # Model will store the generated images used during the evaluation process

    subject_cfgs = [
        { 'name': 'backpack', 'token_identifier': 'sks backpack', 'living': False },
        { 'name': 'backpack_dog', 'token_identifier': 'sks backpack', 'living': False },
        { 'name': 'bear_plushie', 'token_identifier': 'sks stuffed animal', 'living': False },
        { 'name': 'berry_bowl', 'token_identifier': 'sks bowl', 'living': False },
        { 'name': 'can', 'token_identifier': 'sks can', 'living': False },
        { 'name': 'candle', 'token_identifier': 'sks candle', 'living': False },
        { 'name': 'cat', 'token_identifier': 'sks cat', 'living': True },
        { 'name': 'cat2', 'token_identifier': 'sks cat', 'living': True },
        { 'name': 'clock', 'token_identifier': 'sks clock', 'living': False },
        { 'name': 'colorful_sneaker', 'token_identifier': 'sks sneaker', 'living': False },
        { 'name': 'dog', 'token_identifier': 'sks dog', 'living': True },
        { 'name': 'dog2', 'token_identifier': 'sks dog', 'living': True },
        { 'name': 'dog3', 'token_identifier': 'sks dog', 'living': True },
        { 'name': 'dog5', 'token_identifier': 'sks dog', 'living': True },
        { 'name': 'dog6', 'token_identifier': 'sks dog', 'living': True },
        { 'name': 'dog7', 'token_identifier': 'sks dog', 'living': True },
        { 'name': 'dog8', 'token_identifier': 'sks dog', 'living': True },
        { 'name': 'duck_toy', 'token_identifier': 'sks toy', 'living': False },
        { 'name': 'fancy_boot', 'token_identifier': 'sks boot', 'living': False },
        { 'name': 'grey_sloth_plushie', 'token_identifier': 'sks stuffed animal', 'living': False },
        { 'name': 'monster_toy', 'token_identifier': 'sks toy', 'living': False },
        { 'name': 'pink_sunglasses', 'token_identifier': 'sks sunglasses', 'living': False },
        { 'name': 'poop_emoji', 'token_identifier': 'sks toy', 'living': False },
        { 'name': 'rc_car', 'token_identifier': 'sks toy', 'living': False },
        { 'name': 'red_cartoon', 'token_identifier': 'sks cartoon', 'living': True },
        { 'name': 'robot_toy', 'token_identifier': 'sks toy', 'living': False },
        { 'name': 'shiny_sneaker', 'token_identifier': 'sks sneaker', 'living': False },
        { 'name': 'teapot', 'token_identifier': 'sks teapot', 'living': False },
        { 'name': 'vase', 'token_identifier': 'sks vase', 'living': False },
        { 'name': 'wolf_plushie', 'token_identifier': 'sks stuffed animal', 'living': False }
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