import os
from pathlib import Path

from ..pipeline.steps import Step

from ..procedures.sky_spec import (_make_combined_image_sky,
                                   extract_spectra)

from ..procedures.procedures_register import (identify_orders,
                                              identify_lines,
                                              find_affine_transform,
                                              transform_wavelength_solutions,
                                              save_orderflat,
                                              update_db)

from ..qa.register_qa import produce_qa

def _test():
    from igrins import get_obsset
    band = "K"
    config_file = Path("../../recipe.config")
    obsset = get_obsset("20190318", band, "SKY",
                        obsids=range(10, 11),
                        frametypes=["-"],
                        config_file=config_file)


def make_combined_image_sky(obsset, bg_subtraction_mode="flat"):
    if bg_subtraction_mode == "none":
        bg_subtraction_mode = None
    final_sky, cards = _make_combined_image_sky(obsset, bg_subtraction_mode)

    from astropy.io.fits import Card
    fits_cards = [Card(k, v) for (k, v, c) in cards]
    obsset.extend_cards(fits_cards)

    hdul = obsset.get_hdul_to_write(([], final_sky))
    obsset.store("combined_sky", data=hdul)

def obsset_produce_qa_plots(obsset):
    produce_qa(obsset)

steps = [Step("Make Combined Sky", make_combined_image_sky,
              bg_subtraction_mode="none"),
         Step("Extract Simple 1d Spectra", extract_spectra),
         Step("Identify Orders", identify_orders),
         Step("Identify Lines", identify_lines),
         Step("Find Affine Transform", find_affine_transform),
         Step("Derive transformed Wvl. Solution",
              transform_wavelength_solutions),
         Step("Save Order-Flats, etc", save_orderflat),
         Step("Produce QA Plots", obsset_produce_qa_plots),
         Step("Update DB", update_db),
]
