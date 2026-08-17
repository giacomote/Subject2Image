class PipelineConfig:
    """
    Set some configuration variables for the personalization pipeline.
    These are used both during the training and the testing processes.
    """

    data_dir = 'data/dreambooth/cat'  # Relative path from project folder
    adaptation_dir = 'adaptation/'
    results_dir = 'results/'

    token_identifier = 'sks cat'
    training_prompt = f'A photo of {token_identifier}'
    generation_prompt = f'A high quality studio photograph of {token_identifier} on the Moon, with the United States flag behind his back'