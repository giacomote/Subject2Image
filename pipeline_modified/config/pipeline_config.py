class PipelineConfig:
    """
    Set some configuration variables for the personalization pipeline.
    These are used both during the training and the testing processes.
    """

    data_dir = 'data/dreambooth/cat'  # Relative path from project folder
    adaptation_dir = 'adaptation/'
    results_dir = 'images_inference/modified'

    token_identifier = 'sks cat'
    training_steps = 800
    training_prompt = f'A photo of {token_identifier}'
    generation_prompt = f'A high quality studio photograph of {token_identifier} sitting on the edge of a ravine, looking down.'