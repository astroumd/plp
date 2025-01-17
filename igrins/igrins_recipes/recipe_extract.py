"""
"""

from __future__ import print_function

from ..procedures.target_spec import (setup_extraction_parameters,
                                      make_combined_images,
                                      estimate_slit_profile,
                                      extract_stellar_spec,
                                      extract_stellar_spec_pp,
                                      extract_extended_spec_ver2,
                                      store_2dspec)

from ..procedures.a0v_flatten import flatten_a0v
from ..procedures.a0v_telluric_pypeit import flatten_a0v as flatten_a0v_pypeit

from ..pipeline.steps import Step


def set_basename_postfix(obsset, basename_postfix):
    # This only applies for the output name
    obsset.set_basename_postfix(basename_postfix)


def _get_do_ab_from_recipe_name(obsset):
    recipe = obsset.recipe_name
    if recipe.endswith("AB"):
        do_ab = True
    elif recipe.endswith("ONOFF"):
        do_ab = False
    else:
        msg = "recipe name is not supported: {}".format(recipe)
        raise ValueError(msg)

    return do_ab


def estimate_slit_profile_stellar(obsset,
                                  x1=800, x2=None,
                                  do_ab="recipe",
                                  slit_profile_mode="1d"):

    do_ab = _get_do_ab_from_recipe_name(obsset)
    if x2 is None:
        x2 = obsset.detector.nx - x1
    estimate_slit_profile(obsset,
                          x1=x1, x2=x2,
                          do_ab=do_ab, slit_profile_mode=slit_profile_mode)


def estimate_slit_profile_extended(obsset,
                                   x1=800, x2=4096-800,
                                   do_ab="recipe",
                                   slit_profile_mode="uniform"):

    do_ab = _get_do_ab_from_recipe_name(obsset)
    estimate_slit_profile(obsset,
                          x1=x1, x2=x2,
                          do_ab=do_ab,
                          slit_profile_mode=slit_profile_mode)


_steps_default = [
    Step("Setup extraction parameters",
         setup_extraction_parameters,
         height_2dspec=0,
         order_range="-1,-1"),
    Step("Set basename-postfix", set_basename_postfix,
         basename_postfix=""),
]


_steps_stellar = [
    Step("Make Combined Images", make_combined_images),
    Step("Estimate slit profile (stellar)",
         estimate_slit_profile_stellar,
         slit_profile_mode="1d"),
    Step("Extract spectra (for stellar)",
         extract_stellar_spec,
         extraction_mode="optimal"),
    Step("Generate Rectified 2d-spec", store_2dspec),
]

steps_stellar = _steps_default + _steps_stellar


_steps_stellar_pp = [
    Step("Extract spectra (PP for stellar)",
         extract_stellar_spec_pp,
         extraction_mode="optimal"),
]


steps_stellar_pp = _steps_default + _steps_stellar_pp


def update_db(obsset):

    obsset.add_to_db("a0v")


steps_a0v = steps_stellar + [Step("Flatten A0V", flatten_a0v),
                             Step("Update db", update_db)
]

steps_a0v_pypeit = steps_stellar + [Step("Flatten A0V (Pypeit)", flatten_a0v_pypeit),
                                    Step("Update db", update_db)
]


_steps_extended = [
    Step("Make Combined Images", make_combined_images,
         allow_no_b_frame=False),
    Step("Estimate slit profile (extended)", estimate_slit_profile_extended,
         slit_profile_mode="uniform"),
    Step("Extract spectra (for extendeded)",
         extract_extended_spec_ver2,
         lacosmic_thresh=0.,
         # extraction_mode="simple",
    ),
    Step("Generate Rectified 2d-spec", store_2dspec)
]


steps_extended = _steps_default + _steps_extended


if __name__ == "__main__":
    pass
