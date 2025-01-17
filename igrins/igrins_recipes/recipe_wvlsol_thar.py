from __future__ import print_function

from ..procedures.sky_spec import extract_spectra_multi

from ..procedures.process_identify_multiline import identify_multiline_thar
from ..procedures.process_wvlsol_volume_fit import volume_fit


from ..procedures.generate_wvlsol_maps import (make_ordermap_slitposmap,
                                               make_slitoffsetmap,
                                               make_wavelength_map)

from ..procedures.process_derive_wvlsol import derive_wvlsol

# update_distortion_db : see below
# update_wvlsol_db : see below

from ..procedures.process_save_wat_header import save_wat_header

from ..pipeline.steps import Step

from .recipe_register_thar import _make_combined_image_thar


def make_combined_image_thar(obsset, bg_subtraction_mode="flat"):
    final_arc, cards = _make_combined_image_thar(obsset, bg_subtraction_mode)

    from astropy.io.fits import Card
    fits_cards = [Card(k, v) for (k, v, c) in cards]
    obsset.extend_cards(fits_cards)

    hdul = obsset.get_hdul_to_write(([], final_arc))
    obsset.store("combined_thar", data=hdul)


def update_distortion_db(obsset):

    obsset.add_to_db("distortion")


def update_wvlsol_db(obsset):

    obsset.add_to_db("wvlsol")


steps = [Step("Make Combined ThAr", make_combined_image_thar),
         Step("Extract spectra-multi", extract_spectra_multi, comb_type='combined_thar'),
         Step("Identify lines in multi-slit (ThAr)", identify_multiline_thar),
         Step("Derive Distortion Solution", volume_fit, fit_type='thar'),
         Step("Make Ordermap/Slitposmap", make_ordermap_slitposmap),
         Step("Make Slitoffset map", make_slitoffsetmap),
         Step("Update distortion db", update_distortion_db),
         Step("Derive wvlsol", derive_wvlsol, fit_type='thar'),
         Step("Update wvlsol db", update_wvlsol_db),
         Step("Make wvlmap", make_wavelength_map),
         Step("Save WAT header", save_wat_header),
]


if __name__ == "__main__":
    pass
