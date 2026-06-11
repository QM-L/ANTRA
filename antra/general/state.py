from antra.general.config import load_configs

class State():
    '''This class tracks all the user actions and determines whether or not 
       certain actions are currently allowed or not, passes arguments, etc.'''
    def __init__(self):
        # general
        self.page = "setup"

        # setup
        self.dicom = None
        self.segmentations = None
        self.config = load_configs()
        
        # selection
        self.origin = None
        self.tumor_analyzer = None

        # advice
        self.raytracer = None
        self.raytrace_results = None
        self.advice = None
        self.config = load_configs()