import numpy as np

import igrins.libs.ohline_grouped as ohline_grouped

class OHLines(object):
    def __init__(self, fn=None):
        if fn is None:
            fn = "ohlines.dat"
        ohline_ = np.genfromtxt(fn)
        self.um = ohline_[:,0]/1.e4
        self.intensity = ohline_[:,1]/10.
        self._update_wavelengths()

    def _update_wavelengths(self):
        for lines in ohline_grouped.line_groups:
            for l in lines:
                i, wvl = l
                self.um[i] = wvl


if __name__ == "__main__":
    ohlines = OHLines()
