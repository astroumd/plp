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
    nx = 2148
    #print("USING 508x2043 detectors for Deveny")
    #ny0 = 508
    #nx = 2043
    npad_m = 0
    #npad_p = 14
    npad_p = 50
    #npad = npad_p
    ny = ny0 + npad_m + npad_p
    name = 'deveny'

    def update_padding(self, data):
        ny, nx = data.shape()
