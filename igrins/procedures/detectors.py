class Detector(object):
    pass
    #def __init__(self, nx, ny):
    #    self.nx = nx
    #    self.ny = ny

class IGRINSDetector(Detector):
    nx = 2048
    ny = 2048

class RIMASDetector(Detector):
    nx = 4096
    ny = 4096

class RIMASH2RGDetector(Detector):
    nx = 2048
    ny = 2048

class DEVENYDetector(Detector):
    nx = 2148
    ny = 516
