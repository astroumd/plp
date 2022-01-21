import numpy as np

class ThArLines(object):
    def __init__(self, fn):
        tharline_ = np.genfromtxt(fn)
        self.um = tharline_[:, 0]/1.e4
        self.intensity = tharline_[:, 1]/10.

    def _update_wavelengths(self):
        for lines in tharline_grouped.line_groups:
            for l in lines:
                i, wvl = l
                self.um[i] = wvl

def load_thar_ref_data(ref_loader):

    ref_tharline_indices_map = ref_loader.load("THAR_INDICES_JSON")

    ref_tharline_indices = ref_tharline_indices_map[ref_loader.band]

    ref_tharline_indices = dict((int(k), v) for k, v \
            in ref_tharline_indices.items())

    fn = ref_loader.query("THARLINES_JSON")
    tharlines = ThArLines(fn)

    r = dict(tharlines_db = tharlines,
             tharline_indices=ref_tharline_indices)

    return r
