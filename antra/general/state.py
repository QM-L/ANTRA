from antra.general.config import load_configs

class State():
    '''This class tracks all the user actions and determines whether or not 
       certain actions are currently allowed or not, passes arguments, etc.'''
    def __init__(self):
        # general
        self.page = "setup"
        self.config = load_configs()

        # setup
        self.dicom = None
        self.segmentations = None
        self.visualizer = None

        # selection
        self.origin = None
        self.tumor_analyzer = None

        # raytracing
        self.raytracer = None
        self.score_data = None

        # weighing
        self.weights = self.config.gettuple("scoring","weights")
        self.weighted_scores = None

        # advice
        self.advice = None