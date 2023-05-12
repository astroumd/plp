import os
from pathlib import Path

from ..pipeline.steps import Step

from ..procedures.sky_spec import extract_spectra, get_combined_image

from ..procedures.procedures_register import (identify_orders,
                                              identify_lines,
                                              find_affine_transform,
                                              transform_wavelength_solutions,
                                              save_orderflat,
                                              update_db)



def _test():
    from igrins import get_obsset
    band = "K"
    config_file = Path("../../recipe.config")
    obsset = get_obsset("20190318", band, "THAR",
                        obsids=range(10, 11),
                        frametypes=["-"],
                        config_file=config_file)


def make_combined_image_thar(obsset, bg_subtraction_mode="flat"):
    if bg_subtraction_mode == "none":
        bg_subtraction_mode = None
    final_arc, cards = _make_combined_image_thar(obsset, bg_subtraction_mode)

    from astropy.io.fits import Card
    fits_cards = [Card(k, v) for (k, v, c) in cards]
    obsset.extend_cards(fits_cards)

    hdul = obsset.get_hdul_to_write(([], final_arc))
    obsset.store("combined_thar", data=hdul)

def _make_combined_image_thar(obsset, bg_subtraction_mode="flat"):
    thar_data, cards = get_combined_image(obsset)

    thar_data = thar_data / len(obsset.get_obsids())

    return thar_data, cards

steps = [Step("Make Combined ThAr", make_combined_image_thar,
              bg_subtraction_mode="none"),
         Step("Extract Simple 1d Spectra", extract_spectra, comb_type="combined_thar"),
         Step("Identify Orders", identify_orders),
         Step("Identify Lines", identify_lines),
         Step("Find Affine Transform", find_affine_transform),
         Step("Derive transformed Wvl. Solution",
              transform_wavelength_solutions),
         Step("Save Order-Flats, etc", save_orderflat),
         Step("Update DB", update_db),
]
