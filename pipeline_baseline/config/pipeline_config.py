class PipelineConfig:
    """
    Set some configuration variables for the personalization pipeline.
    These are used both during the training and the testing processes.
    """

    data_dir = 'data/cat'  # Relative path from project folder
    adaptation_dir = 'adaptation/'
    results_dir = 'images_inference/baseline'

    token_identifier = '<sks> cat'
    training_prompt = 'A photo of {0}'
    generation_prompt = 'A high quality studio photograph of {0} on the Moon, with the United States flag behind his back'