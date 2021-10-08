class Detector(object):
    pass

class IGRINSDetector(Detector):
    nx = 2048
    ny = 2048

class RIMASDetector(Detector):
    print("RIMAS DETECTOR AT 4096")
    nx = 4096
    ny = 4096
    #print("RIMAS DETECTOR AT 2048")
    #nx = 2048
    #ny = 2048

class RIMASH2RGDetector(Detector):
    nx = 2048
    ny = 2048
