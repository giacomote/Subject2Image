class PipelineConfig:
    """
    Set some configuration variables for the personalization pipeline.
    These are used both during the training and the testing processes.
    """

    data_dir = 'data/cat'  # Relative path from project folder
    adaptation_dir = 'adaptation/'
    results_dir = 'images_inference/modified'

    placeholder_token = '<sks>'
    class_token = 'cat'

    # Textual Inversion configuration: [1, 2] for two text encoders, [1, 2, 3] for three text encoders
    ti_text_encoders = [1, 2, 3]

    training_steps = 800
    training_prompt = 'A photo of {0}'
    generation_prompt = 'A high quality studio photograph of {0} on the Moon, with the United States flag behind his back'