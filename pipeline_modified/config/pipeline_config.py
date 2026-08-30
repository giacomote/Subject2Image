class PipelineConfig:
    """
    Set some configuration variables for the personalization pipeline.
    These are used both during the training and the testing processes.
    """

    data_dir = 'data/dreambooth/cat'  # Relative path from project folder
    adaptation_dir = 'adaptation/'
    results_dir = 'images_inference/modified'

    placeholder_token = 'sks'
    class_token = 'cat'
    token_identifier = f'{placeholder_token} {class_token}'

    # Textual Inversion configuration: [1, 2] for two text encoders, [1, 2, 3] for three text encoders
    ti_text_encoders = [1, 2]

    training_steps = 800
    training_prompt = f'A photo of {token_identifier}'
    generation_prompt = f'A high quality studio photograph of {token_identifier} sitting on the edge of a ravine, looking down.'