import numpy as np
import pandas as pd

from collections import namedtuple
# from igrins.libs.recipe_helper import RecipeHelper


from ..procedures.ref_lines_db import SkyLinesDB, HitranSkyLinesDB, ThArLinesDB

Spec = namedtuple("Spec", ["s_map", "wvl_map", "domain"])


def identify_lines_from_spec(orders, spec_data, wvlsol,
                             ref_lines_db, ref_lines_db_hitrans,
                             ref_sigma=1.5,
                             domains=None):
    small_list = []
    small_keys = []

    spec = Spec(dict(zip(orders, spec_data)),
                dict(zip(orders, wvlsol)),
                dict(zip(orders, domains)))

    fitted_pixels_oh = ref_lines_db.identify(spec, ref_sigma=ref_sigma)
    small_list.append(fitted_pixels_oh)
    small_keys.append("OH")

    # if obsset.band == "K":
    if ref_lines_db_hitrans is not None:
        fitted_pixels_hitran = ref_lines_db_hitrans.identify(spec)
        small_list.append(fitted_pixels_hitran)
        small_keys.append("Hitran")

    fitted_pixels = pd.concat(small_list,
                              keys=small_keys,
                              names=["kind"],
                              axis=0)

    return fitted_pixels

def identify_multiline_thar(obsset):

    multi_spec = obsset.load("multi_spec_fits")

    # just to retrieve order information
    wvlsol_v0 = obsset.load_resource_for("wvlsol_v0")
    orders = wvlsol_v0["orders"]
    wvlsol = wvlsol_v0["wvl_sol"]

    ref_lines_db = ThArLinesDB(obsset.rs.master_ref_loader)
    ref_lines_db_hitrans = None
    
    keys = []
    fitted_pixels_list = []

    for hdu in multi_spec:
        slit_center = hdu.header["FSLIT_CN"]
        keys.append(slit_center)
        print("SLIT CENTER:", slit_center)

        str_test = str(orders[0]) + '_LO'
        if str_test in hdu.header:
            domains = []
            for order in orders:
                str_lo = str(order) + '_LO'
                str_hi = str(order) + '_HI'
                domains.append([int(hdu.header[str_lo]), int(hdu.header[str_hi])])

        import matplotlib.pyplot as plt
        dom = domains[15]
        xvals = np.arange(dom[0],dom[1]+1)
        npts = dom[1] - dom[0] + 1
        plt.figure('FULL SPEC')
        plt.plot(xvals, hdu.data[15, :npts])

        fitted_pixels_ = identify_lines_from_spec(orders, hdu.data, wvlsol,
                                                  ref_lines_db,
                                                  ref_lines_db_hitrans,
                                                  domains=domains)
        
        fitted_pixels_list.append(fitted_pixels_)

    import matplotlib.pyplot as plt
    plt.show()

    # concatenate collected list of fitted pixels.
    fitted_pixels_master = pd.concat(fitted_pixels_list,
                                     keys=keys,
                                     names=["slit_center"],
                                     axis=0)

    # storing multi-index seems broken. Enforce reindexing.
    _d = fitted_pixels_master.reset_index().to_dict(orient="split")

    obsset.store("THAR_FITTED_PIXELS_JSON", _d)


def identify_multiline(obsset):

    multi_spec = obsset.load("multi_spec_fits")

    # just to retrieve order information
    wvlsol_v0 = obsset.load_resource_for("wvlsol_v0")
    orders = wvlsol_v0["orders"]
    wvlsol = wvlsol_v0["wvl_sol"]

    ref_lines_db = SkyLinesDB(obsset.rs.master_ref_loader)

    if obsset.rs.get_resource_spec()[1] == "K":
        ref_lines_db_hitrans = HitranSkyLinesDB(obsset.rs.master_ref_loader)
    else:
        ref_lines_db_hitrans = None

    keys = []
    fitted_pixels_list = []

    for hdu in multi_spec:
        slit_center = hdu.header["FSLIT_CN"]
        keys.append(slit_center)

        str_test = str(orders[0]) + '_LO'
        if str_test in hdu.header:
            domains = []
            for order in orders:
                str_lo = str(order) + '_LO'
                str_hi = str(order) + '_HI'
                domains.append([int(hdu.header[str_lo]), int(hdu.header[str_hi])])

        fitted_pixels_ = identify_lines_from_spec(orders, hdu.data, wvlsol,
                                                  ref_lines_db,
                                                  ref_lines_db_hitrans,
                                                  domains=domains)

        fitted_pixels_list.append(fitted_pixels_)

    # concatenate collected list of fitted pixels.
    fitted_pixels_master = pd.concat(fitted_pixels_list,
                                     keys=keys,
                                     names=["slit_center"],
                                     axis=0)

    # storing multi-index seems broken. Enforce reindexing.
    _d = fitted_pixels_master.reset_index().to_dict(orient="split")
    obsset.store("SKY_FITTED_PIXELS_JSON", _d)


def process_band(utdate, recipe_name, band, obsids, config_name):

    helper = RecipeHelper(config_name, utdate, recipe_name)

    identify_multiline(helper, band, obsids)

