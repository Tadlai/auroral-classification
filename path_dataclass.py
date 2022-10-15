from dataclasses import dataclass


@dataclass
class Paths:
    """
    Specify here all the default paths for saving/loading the necessary files.
    """
    current_preprocessor: str = None
    current_extractor: str = None
    current_reducer: str = None
    current_grouper: str = None
    # Location of raw data:
    root: str = "/home/vincent/DeepLearning/Season_2019-2020"
    # Locations where the trained models along with their parameters will be saved
    machines: str = "/home/vincent/machines"
    preprocessors: str = "/home/vincent/machines/preprocessors"
    extractors: str = "/home/vincent/machines/extractors"
    reducers: str = "/home/vincent/machines/reducers"
    groupers: str = "/home/vincent/machines/groupers"
    # Location where all data outputs will be saved
    data: str = "/home/vincent/data"
    # Location of the manually labelled reference images
    ref: str = "/home/vincent/AuroraShapes/AuroraClasses"
