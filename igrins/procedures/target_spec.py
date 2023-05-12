import numpy as np
import scipy.ndimage as ni

from ..utils.image_combine import image_median
from ..igrins_libs.resource_helper_igrins import ResourceHelper
from ..igrins_recipes.recipe_combine import (make_combined_images
                                             as _make_combined_images)


def _get_int_from_config(obsset, kind, default):
    v = obsset.rs.query_ref_value_from_section("EXTRACTION",
                                               kind,
                                               default=default)
    if v is not None:
        v = int(v)

    return v


def setup_extraction_parameters(obsset, order_range="-1,-1",
                                height_2dspec=0):

    _order_range_s = order_range
    try:
        order_start, order_end = map(int, _order_range_s.split(","))
    except Exception:
        msg = "Failed to parse order range: {}".format(_order_range_s)
        raise ValueError(msg)

    order_start = _get_int_from_config(obsset, "ORDER_START", order_start)
    order_end = _get_int_from_config(obsset, "ORDER_END", order_end)

    height_2dspec = _get_int_from_config(obsset, "HEIGHT_2DSPEC",
                                         height_2dspec)

    obsset.set_recipe_parameters(order_start=order_start,
                                 order_end=order_end,
                                 height_2dspec=height_2dspec)


def _get_combined_image(obsset):
    # Should not use median, Use sum.
    data_list = [hdu.data for hdu in obsset.get_hdus()]

    return np.sum(data_list, axis=0)
    #return image_median(data_list)


def get_destriped(obsset,
                  data_minus,
                  destripe_pattern=64,
                  use_destripe_mask=None,
                  sub_horizontal_median=True,
                  remove_vertical=False):

    from .destriper import destriper

    if use_destripe_mask:
        helper = ResourceHelper(obsset)
        _destripe_mask = helper.get("destripe_mask")

        destrip_mask = ~np.isfinite(data_minus) | _destripe_mask
    else:
        destrip_mask = None

    data_minus_d = destriper.get_destriped(data_minus,
                                           destrip_mask,
                                           pattern=destripe_pattern,
                                           hori=sub_horizontal_median,
                                           remove_vertical=remove_vertical)

    return data_minus_d


def get_variance_map(obsset, data_minus, data_plus):
    helper = ResourceHelper(obsset)
    _destripe_mask = helper.get("destripe_mask")

    bias_mask2 = ni.binary_dilation(_destripe_mask)

    from .variance_map import (get_variance_map,
                               get_variance_map0)

    _pix_mask = helper.get("badpix_mask")
    variance_map0 = get_variance_map0(data_minus,
                                      bias_mask2, _pix_mask)

    _gain = obsset.rs.query_ref_value("GAIN")
    variance_map = get_variance_map(data_plus, variance_map0,
                                    gain=float(_gain))

    return variance_map0, variance_map


def get_variance_map_deprecated(obsset, data_minus, data_plus):
    helper = ResourceHelper(obsset)
    _destripe_mask = helper.get("destripe_mask")

    bias_mask2 = ni.binary_dilation(_destripe_mask)

    from .variance_map import (get_variance_map,
                               get_variance_map0)

    _pix_mask = helper.get("badpix_mask")
    variance_map0 = get_variance_map0(data_minus,
                                      bias_mask2, _pix_mask)

    _gain = obsset.rs.query_ref_value("GAIN")
    variance_map = get_variance_map(data_plus, variance_map0,
                                    gain=float(_gain))

    # variance_map0 : variance without poisson noise of source + sky
    # This is used to estimate model variance where poisson noise is
    # added from the simulated spectra.
    # variance : variance with poisson noise.
    return variance_map0, variance_map


def make_combined_images(obsset,
                         allow_no_b_frame=False,
                         force_image_combine=False):

    try:
        obsset.load("combined_image1")
        combined_image_exists = True
    except Exception:
        combined_image_exists = False
        pass

    if combined_image_exists and not force_image_combine:
        print("skipped")
        return

    _make_combined_images(obsset, allow_no_b_frame=allow_no_b_frame,
                          cache_only=True)


def subtract_interorder_background(obsset, di=24, min_pixel=40):

    data_minus = obsset.load_fits_sci_hdu("COMBINED_IMAGE1").data

    helper = ResourceHelper(obsset)
    sky_mask = helper.get("sky_mask")

    from .estimate_sky import (estimate_background,
                               get_interpolated_cubic)

    xc, yc, v, std = estimate_background(data_minus, sky_mask,
                                         di=di, min_pixel=min_pixel)

    nx = obsset.detector.nx
    ny = nx
    if len(data_minus) != nx:
        raise ValueError("Detector size does not equal size of image:", nx, len(data_minus))

    ZI3 = get_interpolated_cubic(nx, ny, xc, yc, v)

    hdul = obsset.get_hdul_to_write(([], ZI3))
    obsset.store("interorder_background", data=hdul, cache_only=True)

    hdul = obsset.get_hdul_to_write(([], ZI3))
    obsset.store("combined_image1", data=hdul, cache_only=True)


def estimate_slit_profile(obsset,
                          x1=800, x2=None,
                          do_ab=True, slit_profile_mode="1d"):

    #TODO: Check to see what I need to change for RIMAS
    #NJM Might need to change x-range of slits (aka increase default value for x1???)
    if x2 is None:
        x2 = obsset.detector.nx - x1

    if slit_profile_mode == "1d":
        from .slit_profile import estimate_slit_profile_1d
        estimate_slit_profile_1d(obsset, x1=x1, x2=x2, do_ab=do_ab)
    elif slit_profile_mode == "uniform":
        from .slit_profile import estimate_slit_profile_uniform
        estimate_slit_profile_uniform(obsset, do_ab=do_ab)
    else:
        msg = ("Unknwon mode ({}) in slit_profile estimation"
               .format(slit_profile_mode))
        raise ValueError(msg)


def get_wvl_header_data(obsset, wavelength_increasing_order=False):
    # from ..libs.storage_descriptions import SKY_WVLSOL_FITS_DESC
    # fn = igr_storage.get_path(SKY_WVLSOL_FITS_DESC,
    #                           extractor.basenames["wvlsol"])
    # fn = sky_path.get_secondary_path("wvlsol_v1.fits")
    # f = pyfits.open(fn)

    hdu = obsset.load_resource_sci_hdu_for("wvlsol_fits")
    if wavelength_increasing_order:
        from ..utils import iraf_helper
        header = iraf_helper.invert_order(hdu.header)

        def convert_data(d):
            return d[::-1]
    else:
        header = hdu.header

        def convert_data(d):
            return d

    return header.copy(), hdu.data, convert_data


def store_1dspec(obsset, v_list, s_list, sn_list=None, domain_list=None, first_order=0):

    basename_postfix = obsset.basename_postfix

    wvl_header, wvl_data, convert_data = get_wvl_header_data(obsset)

    def add_domain_header(hdul, domain_list):
        for i, domain in enumerate(domain_list):
            str_lo = str(i) + '_LO'
            str_hi = str(i) + '_HI'
            hdul[0].header[str_lo] = domain[0]
            hdul[0].header[str_hi] = domain[1]
        hdul[0].header['ORDER0'] = first_order
        return hdul

    if domain_list is not None:
        #spectra are different sizes for each order. Must pad
        #spectra so they are all the same size
        max_val = 0
        for s in s_list:
            max_val = max(max_val, len(s))
        d = np.zeros([len(v_list), max_val])
        for i, v in enumerate(v_list):
            n = len(v)
            d[i, :n] = v
    else:
        d = np.array(v_list)
    v_data = convert_data(d.astype("float32"))

    hdul = obsset.get_hdul_to_write(([], v_data))
    wvl_header.update(hdul[0].header)
    hdul[0].header = wvl_header
    if domain_list is not None:
        hdul = add_domain_header(hdul, domain_list)
    hdul[0].verify(option="silentfix")

    obsset.store("VARIANCE_FITS", hdul,
                 postfix=basename_postfix)

    if sn_list is not None:
        if domain_list is not None:
            d = np.zeros([len(sn_list), max_val])
            for i, sn in enumerate(sn_list):
                n = len(sn)
                d[i, :n] = sn
        else:
            d = np.array(sn_list)
        sn_data = convert_data(d.astype("float32"))

        hdul = obsset.get_hdul_to_write(([], sn_data))
        wvl_header.update(hdul[0].header)
        hdul[0].header = wvl_header
        if domain_list is not None:
            hdul = add_domain_header(hdul, domain_list)
        obsset.store("SN_FITS", hdul,
                     postfix=basename_postfix)

    if domain_list is not None:
        d = np.zeros([len(s_list), max_val])
        for i, s in enumerate(s_list):
            n = len(s)
            d[i, :n] = s
    else:
        d = np.array(s_list)
    s_data = convert_data(d.astype("float32"))

    hdul = obsset.get_hdul_to_write(([], s_data),
                                    ([], convert_data(wvl_data)))
    wvl_header.update(hdul[0].header)
    hdul[0].header = wvl_header
    if domain_list is not None:
        hdul = add_domain_header(hdul, domain_list)
    hdul[0].verify(option="silentfix")

    obsset.store("SPEC_FITS", hdul,
                 postfix=basename_postfix)


def store_2dspec(obsset,
                 conserve_flux=True):

    basename_postfix = obsset.basename_postfix

    height_2dspec = obsset.get_recipe_parameter("height_2dspec")

    from .shifted_images import ShiftedImages
    hdul = obsset.load("WVLCOR_IMAGE", postfix=basename_postfix)
    shifted = ShiftedImages.from_hdul(hdul)

    data_shft = shifted.image
    variance_map_shft = shifted.variance

    wvl_header, wvl_data, convert_data = get_wvl_header_data(obsset)

    bottom_up_solutions_ = obsset.load_resource_for("aperture_definition")
    bottom_up_solutions = bottom_up_solutions_["bottom_up_solutions"]
    domain_list = bottom_up_solutions_["domain"]

    helper = ResourceHelper(obsset)
    ordermap_bpixed = helper.get("ordermap_bpixed")

    from .correct_distortion import get_rectified_2dspec
    _ = get_rectified_2dspec(data_shft,
                             ordermap_bpixed,
                             bottom_up_solutions,
                             conserve_flux=conserve_flux,
                             height=height_2dspec,
                             domain=domain_list)
    d0_shft_list, msk_shft_list = _

    with np.errstate(invalid="ignore"):
        d = np.array(d0_shft_list) / np.array(msk_shft_list)

    hdul = obsset.get_hdul_to_write(([], convert_data(d.astype("float32"))))
    # wvl_header.update(hdul[0].header)
    hdul[0].header = wvl_header

    obsset.store("SPEC2D_FITS", hdul, postfix=basename_postfix)

    # OUTPUT VAR2D, added by Kyle Kaplan Feb 25, 2015 to get variance map
    # outputted as a datacube
    _ = get_rectified_2dspec(variance_map_shft,
                             ordermap_bpixed,
                             bottom_up_solutions,
                             conserve_flux=conserve_flux,
                             height=height_2dspec,
                             domain=domain_list)
    d0_shft_list, msk_shft_list = _

    with np.errstate(invalid="ignore"):
        d = np.array(d0_shft_list) / np.array(msk_shft_list)

    hdul = obsset.get_hdul_to_write(([], convert_data(d.astype("float32"))))
    # wvl_header.update(hdul[0].header)
    hdul[0].header = wvl_header

    obsset.store("VAR2D_FITS", hdul, postfix=basename_postfix)


def extract_stellar_spec(obsset, extraction_mode="optimal",
                         conserve_2d_flux=True, calculate_sn=True):

    # refactored from recipe_extract.ProcessABBABand.process

    helper = ResourceHelper(obsset)

    ap = helper.get("aperture")

    postfix = obsset.basename_postfix

    #Load all the maps that used in the spectra extraction
    data_minus = obsset.load_fits_sci_hdu("COMBINED_IMAGE1",
                                          postfix=postfix).data

    orderflat = helper.get("orderflat")
    data_minus_flattened = data_minus / orderflat

    variance_map = obsset.load_fits_sci_hdu("combined_variance1",
                                            postfix=postfix).data
    variance_map0 = obsset.load_fits_sci_hdu("combined_variance0",
                                             postfix=postfix).data

    slitoffset_map = helper.get("slitoffsetmap")

    ordermap = helper.get("ordermap")
    ordermap_bpixed = helper.get("ordermap_bpixed")
    slitpos_map = helper.get("slitposmap")

    gain = float(obsset.rs.query_ref_value("gain"))

    profile_map = obsset.load_fits_sci_hdu("slitprofile_fits",
                                           postfix=postfix).data

    #Extract spectra using the calculated 2D profile and the loaded
    #data files
    from .spec_extract_w_profile import extract_spec_using_profile
    _ = extract_spec_using_profile(ap, profile_map,
                                   variance_map,
                                   variance_map0,
                                   data_minus_flattened,
                                   orderflat,
                                   ordermap, ordermap_bpixed,
                                   slitpos_map,
                                   slitoffset_map,
                                   gain,
                                   extraction_mode=extraction_mode,
                                   debug=False)

    s_list, v_list, cr_mask, aux_images = _

    if calculate_sn:
        # calculate S/N per resolution
        wvl_solutions = helper.get("wvl_solutions")

        key_list = []
        domain_list = []
        for key in ap.domain_dict:
            key_list.append(key)
            domain_list.append(ap.domain_dict[key])
        idx_sort = np.argsort(key_list)
        domain_list_sort = []
        for idx in idx_sort:
            domain_list_sort.append(domain_list[idx])

        sn_list = []
        for wvl, s, v, domain in zip(wvl_solutions,
                                     s_list, v_list,
                                     domain_list_sort):

            wvl = wvl[domain[0]:domain[1]+1]
            dw = np.gradient(wvl)
            pixel_per_res_element = (wvl/40000.)/dw

            with np.errstate(invalid="ignore"):
                sn = (s/v**.5)*(pixel_per_res_element**.5)

            sn_list.append(sn)
    else:
        sn_list = None

    first_order = key_list[0]
    store_1dspec(obsset, v_list, s_list, sn_list=sn_list,
                 domain_list=domain_list_sort, first_order=first_order)

    hdul = obsset.get_hdul_to_write(([], data_minus),
                                    ([], aux_images["synth_map"]))
    obsset.store("DEBUG_IMAGE", hdul)

    shifted = aux_images["shifted"]

    _hdul = shifted.to_hdul()
    hdul = obsset.get_hdul_to_write(*_hdul)
    obsset.store("WVLCOR_IMAGE", hdul)

    # store_2dspec(obsset,
    #              shifted.image,
    #              shifted.variance,
    #              ordermap_bpixed,
    #              cr_mask=cr_mask,
    #              conserve_flux=conserve_2d_flux,
    #              height_2dspec=height_2dspec)


def extract_stellar_spec_pp(obsset, extraction_mode="optimal", height_2dspec=0,
                            conserve_2d_flux=True, calculate_sn=True):

    # refactored from recipe_extract.ProcessABBABand.process

    helper = ResourceHelper(obsset)

    ap = helper.get("aperture")

    postfix = obsset.basename_postfix
    # data_minus = obsset.load_fits_sci_hdu("COMBINED_IMAGE1",
    #                                       postfix=postfix).data

    # orderflat = helper.get("orderflat")
    # data_minus_flattened = data_minus / orderflat

    # variance_map = obsset.load_fits_sci_hdu("combined_variance1",
    #                                         postfix=postfix).data
    # variance_map0 = obsset.load_fits_sci_hdu("combined_variance0",
    #                                          postfix=postfix).data

    # slitoffset_map = helper.get("slitoffsetmap")

    ordermap = helper.get("ordermap")
    ordermap_bpixed = helper.get("ordermap_bpixed")
    # slitpos_map = helper.get("slitposmap")

    # # from .slit_profile import get_profile_func
    # # profile = get_profile_func(obsset)

    # gain = float(obsset.rs.query_ref_value("gain"))

    # profile_map = obsset.load_fits_sci_hdu("slitprofile_fits",
    #                                        postfix=postfix).data

    from .shifted_images import ShiftedImages
    hdul = obsset.load("WVLCOR_IMAGE", postfix=postfix)
    shifted = ShiftedImages.from_hdul(hdul)

    from .spec_extract_from_shifted import extract_spec_from_shifted
    _ = extract_spec_from_shifted(ap,
                                  ordermap, ordermap_bpixed,
                                  shifted,
                                  extraction_mode=extraction_mode,
                                  debug=False)

    s_list, v_list = _

    if calculate_sn:
        # calculate S/N per resolution
        wvl_solutions = helper.get("wvl_solutions")

        sn_list = []
        for wvl, s, v in zip(wvl_solutions,
                             s_list, v_list):

            dw = np.gradient(wvl)
            pixel_per_res_element = (wvl/40000.)/dw

            with np.errstate(invalid="ignore"):
                sn = (s/v**.5)*(pixel_per_res_element**.5)

            sn_list.append(sn)
    else:
        sn_list = None

    store_1dspec(obsset, v_list, s_list, sn_list=sn_list)

    # hdul = obsset.get_hdul_to_write(([], data_minus),
    #                                 ([], aux_images["synth_map"]))
    # obsset.store("DEBUG_IMAGE", hdul)

    # shifted = aux_images["shifted"]

    # _hdul = shifted.to_hdul()
    # hdul = obsset.get_hdul_to_write(*_hdul)
    # obsset.store("WVLCOR_IMAGE", hdul)


def extract_extended_spec1(obsset, data,
                           variance_map, variance_map0,
                           lacosmic_thresh=0.):

    # refactored from recipe_extract.ProcessABBABand.process

    helper = ResourceHelper(obsset)

    ap = helper.get("aperture")

    orderflat = helper.get("orderflat")

    data_minus = data
    data_minus_flattened = data_minus / orderflat

    slitoffset_map = helper.get("slitoffsetmap")

    ordermap = helper.get("ordermap")
    ordermap_bpixed = helper.get("ordermap_bpixed")
    slitpos_map = helper.get("slitposmap")

    wvl_solutions = helper.get("wvl_solutions")

    # from .slit_profile import get_profile_func
    # profile = get_profile_func(obsset)

    gain = float(obsset.rs.query_ref_value("gain"))

    postfix = obsset.basename_postfix
    profile_map = obsset.load_fits_sci_hdu("slitprofile_fits",
                                           postfix=postfix).data

    from .spec_extract_w_profile import extract_spec_uniform
    _ = extract_spec_uniform(ap, profile_map,
                             variance_map,
                             variance_map0,
                             data_minus_flattened,
                             data_minus, orderflat,  #
                             ordermap, ordermap_bpixed,
                             slitpos_map,
                             slitoffset_map,
                             gain,
                             lacosmic_thresh=lacosmic_thresh,
                             debug=False)

    s_list, v_list, cr_mask, aux_images = _

    return s_list, v_list, cr_mask, aux_images

def extract_extended_spec(obsset, lacosmic_thresh=0., calculate_sn=True):

    # refactored from recipe_extract.ProcessABBABand.process
    
    #TODO: Refactor to match extract_stellar_spec
    helper = ResourceHelper(obsset)
    ap = helper.get("aperture")

    from ..utils.load_fits import get_science_hdus
    postfix = obsset.basename_postfix
    hdul = get_science_hdus(obsset.load("COMBINED_IMAGE1",
                                        postfix=postfix))
     
    data = hdul[0].data

    if len(hdul) == 3:
        variance_map = hdul[1].data
        variance_map0 = hdul[2].data
    else:
        variance_map = obsset.load_fits_sci_hdu("combined_variance1",
                                                postfix=postfix).data
        variance_map0 = obsset.load_fits_sci_hdu("combined_variance0",
                                                 postfix=postfix).data

    _ = extract_extended_spec1(obsset, data,
                               variance_map, variance_map0,
                               lacosmic_thresh=lacosmic_thresh)

    s_list, v_list, cr_mask, aux_images = _

    if calculate_sn:
        # calculate S/N per resolution
        helper = ResourceHelper(obsset)
        wvl_solutions = helper.get("wvl_solutions")
        
        key_list = []
        domain_list = []
        for key in ap.domain_dict:
            key_list.append(key)
            domain_list.append(ap.domain_dict[key])
        idx_sort = np.argsort(key_list)
        domain_list_sort = []
        for idx in idx_sort:
            domain_list_sort.append(domain_list[idx])
        #domain_list_sort = domain_list[idx_sort]

        sn_list = []
        for wvl, s, v, domain in zip(wvl_solutions,
                                     s_list, v_list,
                                     domain_list_sort):

            wvl = wvl[domain[0]:domain[1]+1]
            dw = np.gradient(wvl)
            pixel_per_res_element = (wvl/40000.)/dw

            sn = (s/v**.5)*(pixel_per_res_element**.5)

            sn_list.append(sn)

    first_order = key_list[0]
    store_1dspec(obsset, v_list, s_list, sn_list=sn_list,
                 domain_list=domain_list_sort, first_order=first_order)

    shifted = aux_images["shifted"]

    _hdul = shifted.to_hdul()
    hdul = obsset.get_hdul_to_write(*_hdul)
    obsset.store("WVLCOR_IMAGE", hdul, postfix=obsset.basename_postfix)

    # store_2dspec(obsset,
    #              shifted.image,
    #              shifted.variance_map,
    #              ordermap_bpixed,
    #              cr_mask=cr_mask,
    #              conserve_flux=conserve_2d_flux,
    #              height_2dspec=height_2dspec)

def extract_extended_spec_ver2(obsset, lacosmic_thresh=0., calculate_sn=True):
    '''Matches extract_stellar_spec except for calling the different function
    in spec_extract_w_profile
    '''

    helper = ResourceHelper(obsset)
    ap = helper.get("aperture")

    postfix = obsset.basename_postfix

    #Load the data needed to extract the spectra
    data_minus = obsset.load_fits_sci_hdu("COMBINED_IMAGE1",
                                          postfix=postfix).data

    orderflat = helper.get("orderflat")
    data_minus_flattened = data_minus / orderflat

    variance_map = obsset.load_fits_sci_hdu("combined_variance1",
                                            postfix=postfix).data
    variance_map0 = obsset.load_fits_sci_hdu("combined_variance0",
                                             postfix=postfix).data

    slitoffset_map = helper.get("slitoffsetmap")

    ordermap = helper.get("ordermap")
    ordermap_bpixed = helper.get("ordermap_bpixed")
    slitpos_map = helper.get("slitposmap")

    gain = float(obsset.rs.query_ref_value("gain"))

    profile_map = obsset.load_fits_sci_hdu("slitprofile_fits",
                                           postfix=postfix).data

    #Extended-an/onoff uses the uniform profile and is the only difference
    #from the stellar-ab/onoff observations
    from .spec_extract_w_profile import extract_spec_uniform
    _ = extract_spec_uniform(ap, profile_map,
                             variance_map,
                             variance_map0,
                             data_minus_flattened,
                             data_minus, orderflat,  #
                             ordermap, ordermap_bpixed,
                             slitpos_map,
                             slitoffset_map,
                             gain,
                             lacosmic_thresh=lacosmic_thresh,
                             debug=False)

    s_list, v_list, cr_mask, aux_images = _

    if calculate_sn:
        wvl_solutions = helper.get("wvl_solutions")

        key_list = []
        domain_list = []
        for key in ap.domain_dict:
            key_list.append(key)
            domain_list.append(ap.domain_dict[key])
        idx_sort = np.argsort(key_list)
        domain_list_sort = []
        for idx in idx_sort:
            domain_list_sort.append(domain_list[idx])

        sn_list = []
        for wvl, s, v, domain in zip(wvl_solutions,
                                     s_list, v_list,
                                     domain_list_sort):

            wvl = wvl[domain[0]:domain[1]+1]
            dw = np.gradient(wvl)
            pixel_per_res_element = (wvl/40000.)/dw

            with np.errstate(invalid="ignore"):
                sn = (s/v**.5)*(pixel_per_res_element**.5)

            sn_list.append(sn)

    first_order = key_list[0]

    store_1dspec(obsset, v_list, s_list, sn_list=sn_list,
                 domain_list=domain_list_sort, first_order=first_order)

    shifted = aux_images["shifted"]

    _hdul = shifted.to_hdul()
    hdul = obsset.get_hdul_to_write(*_hdul)
    obsset.store("WVLCOR_IMAGE", hdul, postfix=obsset.basename_postfix)


def _get_slit_profile_options(slit_profile_options):
    slit_profile_options = slit_profile_options.copy()
    n_comp = slit_profile_options.pop("n_comp", None)
    stddev_list = slit_profile_options.pop("stddev_list", None)
    if slit_profile_options:
        msgs = ["unrecognized options: %s"
                % slit_profile_options,
                "\n",
                "Available options are: n_comp, stddev_list"]

        raise ValueError("".join(msgs))

    return n_comp, stddev_list


