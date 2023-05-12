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
    ny0 = 516
    nx = 2148
    npad_m = 0
    npad_p = 50
    ny = ny0 + npad_m + npad_p
    name = 'deveny'

    def update_padding(self, data):
        ny, nx = data.shape()
