class PipelineConfig:
    """
    Set some configuration variables for the personalization pipeline.
    These are used both during the training and the testing processes.
    """

    data_dir = 'data/dreambooth/cat'  # Relative path from project folder
    adaptation_dir = 'adaptation/'
    results_dir = 'results/'