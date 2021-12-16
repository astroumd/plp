class Detector(object):
    pass
    #def __init__(self, nx, ny):
    #    self.nx = nx
    #    self.ny = ny

class IGRINSDetector(Detector):
    nx = 2048
    ny = 2048
    name = 'igrins'

class RIMASDetector(Detector):
    nx = 4096
    ny = 4096
    name = 'rimas'

class RIMASH2RGDetector(Detector):
    nx = 2048
    ny = 2048
    name = 'rimash2rg'

class DEVENYDetector(Detector):
    print("ADD INFO ABOUT PADDING TO DETECTORS.PY")
    print("DO I NEED TO MODIFY THE FILL OF PADDING")
    ny0 = 516
    npad = 14
    ny = ny0 + npad
    nx = 2148
    name = 'deveny'
    #ny = 516
    #npad = 14
    #ny = 530 #padded by 4
