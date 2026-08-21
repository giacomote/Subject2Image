class PipelineConfig:
    """
    Set some configuration variables for the personalization pipeline.
    These are used both during the training and the testing processes.
    """

    data_dir = 'data/dreambooth/dog5'  # Relative path from project folder
    adaptation_dir = 'adaptation/'
    results_dir = 'results/'

    placeholder_token = 'sks'  # Placeholder token used to represent the specific subject
    class_token = 'dog'  # Class token describing the general category of the subject
    token_identifier = f'{placeholder_token} {class_token}'  # Combined token used to identify the personalized subject

    training_prompt = f'A photo of {token_identifier}'
    generation_prompt = f'A high quality studio photograph of {token_identifier} on a sofa, with a living room in the background'
