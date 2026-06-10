from antra.general.config import load_configs

class State():
    '''This class tracks all the user actions and determines whether or not 
       certain actions are currently allowed or not, passes arguments, etc.'''
    def __init__(self):
        self.page = "setup"
        self.dicom = None
        self.config = load_configs()