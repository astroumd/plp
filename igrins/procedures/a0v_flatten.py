import os

import numpy as np


def _get_a0v_interp1d(obsset):
    """Generate an interpolation function for the input VEGA spectra
    """

    vega_data = obsset.rs.load_ref_data("VEGA_SPEC")
    from .a0v_spec import A0VSpec
    a0v_spec = A0VSpec(vega_data)
    print("REPLACING MIN/MAX WVL IN _GET_A0V_INTERP1D")
    #Original code used 1.3, 2.5. We needed to increase the range
    #because of the larger frequency coverage in these detectors

    wvl1 = 0.3
    wvl2 = 3.0
    if obsset.detector.name.lower() == 'deveny':
        wvl1 = 0.32
        wvl2 = 0.9

    a0v_interp1d = a0v_spec.get_flux_interp1d(wvl1, wvl2,
                                              flatten=False,
                                              smooth_pixel=32)
    return a0v_interp1d


def flatten_a0v(obsset, fill_nan=None):  # refactor of get_a0v_flattened
    "This is the main function to do flattening"

    from ..igrins_libs.resource_helper_igrins import ResourceHelper
    helper = ResourceHelper(obsset)

    wvl_solutions = helper.get("wvl_solutions")  # extractor.wvl_solutionsw
    domain_list = helper.get("domain_list")

    tel_interp1d_f = get_tel_interp1d_f(obsset, wvl_solutions)

    a0v_interp1d = _get_a0v_interp1d(obsset)
    # from ..libs.a0v_spec import A0V
    # a0v_interp1d = A0V.get_flux_interp1d(self.config)
    orderflat_json = obsset.load_resource_for("order_flat_json")
    orderflat_response = orderflat_json["fitted_responses"]

    s_list = obsset.load_fits_sci_hdu("SPEC_FITS").data

    from .a0v_flatten_telluric import get_a0v_flattened
    data_list = get_a0v_flattened(a0v_interp1d, tel_interp1d_f,
                                  wvl_solutions, s_list, orderflat_response,
                                  domain_list)

    if fill_nan is not None:
        flattened_s = data_list[0][1]
        flattened_s[~np.isfinite(flattened_s)] = fill_nan

    store_a0v_results(obsset, data_list)


def store_a0v_results(obsset, a0v_flattened_data):

    from .target_spec import get_wvl_header_data
    wvl_header, wvl_data, convert_data = get_wvl_header_data(obsset)

    image_list = []
    image_list.append(([("EXTNAME", "SPEC_FLATTENED")],
                       convert_data(a0v_flattened_data[0][1])))

    for ext_name, data in a0v_flattened_data[1:]:
        _ = ([("EXTNAME", ext_name.upper())], convert_data(data))
        image_list.append(_)

    hdul = obsset.get_hdul_to_write(*image_list)
    wvl_header.update(hdul[0].header)
    hdul[0].header = wvl_header
    hdul[0].verify(option="silentfix")

    obsset.store("spec_fits_flattened", hdul)


def get_tel_interp1d_f(obsset, wvl_solutions):
    """Generates an interpolation function for the telluric transmission model
    """

    telfit_outname_npy = obsset.rs.query_ref_data_path("TELFIT_MODEL")
    telfit_outname_npy = telfit_outname_npy.replace('/', os.path.sep)
    
    from ..igrins_libs.logger import debug

    debug("loading TELFIT_MODEL: {}".format(telfit_outname_npy))

    from .a0v_flatten_telluric import TelluricTransmission
    tel_trans = TelluricTransmission(telfit_outname_npy)

    wvl_solutions = np.array(wvl_solutions)

    w_min = wvl_solutions.min()*0.9
    w_max = wvl_solutions.max()*1.1

    def tel_interp1d_f(gw=None):
        return tel_trans.get_telluric_trans_interp1d(w_min, w_max, gw)

    return tel_interp1d_f
