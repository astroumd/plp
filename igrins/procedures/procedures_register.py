from __future__ import print_function

import warnings

import numpy as np
import matplotlib  # Affine class is used

from .. import DESCS
from ..igrins_libs.resource_helper_igrins import ResourceHelper
from ..utils.load_fits import get_first_science_hdu

from .aperture_helper import get_simple_aperture_from_obsset


def _get_ref_spec_name(recipe_name):

    if (recipe_name in ["SKY"]) or recipe_name.endswith("_AB"):
        ref_spec_key = "SKY_REFSPEC_JSON"
        ref_identified_lines_key = "SKY_IDENTIFIED_LINES_V0_JSON"

    elif recipe_name in ["THAR"]:
        #ref_spec_key = "ARCS_REFSPEC_JSON"
        #ref_identified_lines_key = "ARCS_IDENTIFIED_LINES_V0_JSON"
        ref_spec_key = "THAR_REFSPEC_JSON"
        ref_identified_lines_key = "THAR_IDENTIFIED_LINES_V0_JSON"

    else:
        raise ValueError("Recipe name of '%s' is unsupported."
                         % recipe_name)

    return ref_spec_key, ref_identified_lines_key


def _match_order(src_spectra, ref_spectra):

    orders_ref = ref_spectra["orders"]
    s_list_ref = ref_spectra["specs"]

    s_list_ = src_spectra["specs"]
    s_list = [np.array(s) for s in s_list_]

    # match the orders of s_list_src & s_list_dst
    from .match_orders import match_orders
    delta_indx, orders = match_orders(orders_ref, s_list_ref,
                                      s_list)

    return orders


def identify_orders(obsset):

    ref_spec_key, _ = _get_ref_spec_name(obsset.recipe_name)

    ref_spec_path, ref_spectra = obsset.rs.load_ref_data(ref_spec_key,
                                                         get_path=True)

    src_spectra = obsset.load(DESCS["ONED_SPEC_JSON"])
    
    dom_src = src_spectra['domain_dict']
    #dom_orders = dom_src.keys()
    dom_values = dom_src.values()
   
    new_orders = _match_order(src_spectra, ref_spectra)

    dom_dict = {}
    for order, dom_value in zip(new_orders, dom_values):
        dom_dict[int(order)] = dom_value

    from ..igrins_libs.logger import info
    info("          orders: {}...{}".format(new_orders[0], new_orders[-1]))

    src_spectra["orders"] = new_orders
    src_spectra["domain_dict"] = dom_dict

    obsset.store(DESCS["ONED_SPEC_JSON"],
                 data=src_spectra)

    aperture_basename = src_spectra["aperture_basename"]
    obsset.store(DESCS["ORDERS_JSON"],
                 data=dict(orders=new_orders,
                           aperture_basename=aperture_basename,
                           ref_spec_path=ref_spec_path))


def _get_offset_transform(thar_spec_src, thar_spec_dst, domains):
    
    def get_offsetter(o):
        def _f(x, o=o):
            return x+o
        return _f

    from scipy.signal import correlate
    offsets = []
    cor_list = []
    nxs = []

    offsets_corr = []

    for s_src, s_dst, dom in zip(thar_spec_src, thar_spec_dst, domains):
        nx_src = len(s_src)
        nx_dst = len(s_dst)
        nx = nx_dst
        nxs.append(nx)
        center = nx / 2.

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)

            cor = correlate(s_src, s_dst, mode="same")

        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.plot(cor)
        #plt.figure()
        #plt.plot(s_src)
        #plt.plot(s_dst)
        #plt.show()

        cor_list.append(cor)
        offset = center - np.argmax(cor)
        offsets.append(offset)

        offsets_corr.append(offset + dom[0])

    if len(offsets) == 1:
        offsets2 = [offsets[0]]
        sol_list = [get_offsetter(offset_) for offset_ in offsets2]
        return dict(sol_type="offset",
                    sol_list=sol_list,
                    offsets_orig=offsets,
                    offsets_revised=offsets2)

    from .skimage_measure_fit import ransac, LineModel


    xi = np.arange(len(offsets))
    #data = np.array([xi, offsets]).T
    print("USING OFFSETS AFTER ADDING IN DOMAIN CUTOFF FOR BETTER OUTLIER CALCULATION")
    data = np.array([xi, offsets_corr]).T
    model_robust, inliers = ransac(data,
                                   LineModel, min_samples=3,
                                   residual_threshold=2, max_trials=100)

    outliers_indices = xi[inliers == False]
    offsets2 = [o for o in offsets]
    
    for i in outliers_indices:
        # reduce the search range for correlation peak using the model
        # prediction.
        nx = nxs[i]
        center = nx / 2.
        ym = int(model_robust.predict_y(i))
        x1 = int(max(0, (center - ym) - 20))
        x2 = int(min((center - ym) + 20 + 1, nx))
        ym2 = center - (np.argmax(cor_list[i][x1:x2]) + x1)
        # print ym2
        offsets2[i] = ym2

    sol_list = [get_offsetter(offset_) for offset_ in offsets2]

    return dict(sol_type="offset",
                sol_list=sol_list,
                offsets_orig=offsets,
                offsets_revised=offsets2)


def _get_offset_transform_between_2spec(ref_spec, tgt_spec):

    orders_ref = ref_spec["orders"]
    s_list_ref = ref_spec["specs"]

    orders_tgt = tgt_spec["orders"]
    s_list_tgt = tgt_spec["specs"]
    dom_tgt = tgt_spec["domain_dict"]
    
    s_list_tgt = [np.array(s) for s in s_list_tgt]
    
    orders_intersection = set(orders_ref).intersection(orders_tgt)
    orders_intersection = sorted(orders_intersection)

    def filter_order(orders, s_list, orders_intersection):
        s_dict = dict(zip(orders, s_list))
        s_list_filtered = [s_dict[o] for o in orders_intersection]
        return s_list_filtered

    def filter_domain(orders, domain_dict):
        domain_filtered = [domain_dict[o] for o in orders]
        return domain_filtered

    s_list_ref_filtered = filter_order(orders_ref, s_list_ref,
                                       orders_intersection)
    s_list_tgt_filtered = filter_order(orders_tgt, s_list_tgt,
                                       orders_intersection)
    domains_filt = filter_domain(orders_intersection, dom_tgt)

    offset_transform = _get_offset_transform(s_list_ref_filtered,
                                             s_list_tgt_filtered,
                                             domains_filt)

    return orders_intersection, offset_transform


def identify_lines(obsset):

    _ = _get_ref_spec_name(obsset.recipe_name)
    ref_spec_key, ref_identified_lines_key = _

    ref_spec = obsset.rs.load_ref_data(ref_spec_key)

    tgt_spec = obsset.load(DESCS["ONED_SPEC_JSON"])
    # tgt_spec_path = obsset.query_item_path("ONED_SPEC_JSON")
    # tgt_spec = obsset.load_item("ONED_SPEC_JSON")

    intersected_orders, d = _get_offset_transform_between_2spec(ref_spec,
                                                                tgt_spec)

    #NJM: Remove
    #print("SSS:", np.shape(ref_spec['specs'][0]))
    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.plot(tgt_spec['specs'][0], 'b', label='TGT Spec')
    #plt.plot(ref_spec['specs'][0], 'r', label='Ref Spec')
    ##plt.plot(ref_spec['specs'][0][56:], 'r', label='Ref Spec 56')
    ##plt.plot(ref_spec['specs'][0][57:], 'm', label='Ref Spec 57')
    #plt.legend(loc=0, prop={'size': 12})
    #plt.show()
    ##zzz

    # REF_TYPE="OH"
    # fn = "../%s_IGRINS_identified_%s_%s.json" % (REF_TYPE, band,
    #                                             helper.refdate)
    l = obsset.rs.load_ref_data(ref_identified_lines_key)
    # l = json.load(open(fn))
    # ref_spectra = load_ref_data(helper.config, band, kind="SKY_REFSPEC_JSON")

    offsetfunc_map = dict(zip(intersected_orders, d["sol_list"]))

    from .identified_lines import IdentifiedLines

    identified_lines_ref = IdentifiedLines(l)
    ref_map = identified_lines_ref.get_dict()

    identified_lines_tgt = IdentifiedLines(l)
    identified_lines_tgt.update(dict(wvl_list=[], ref_indices_list=[],
                                     pixpos_list=[], orders=[],
                                     groupname=obsset.groupname))

    from .line_identify_simple import match_lines1_pix

    x0 = 0
    for o, s in zip(tgt_spec["orders"], tgt_spec["specs"]):
        if (o not in ref_map) or (o not in offsetfunc_map):
            wvl, indices, pixpos = [], [], []
        else:
            pixpos, indices, wvl = ref_map[o]
            pixpos = np.array(pixpos)
            msk = (pixpos >= 0)

            ref_pix_list = offsetfunc_map[o](pixpos[msk])
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', r'Degrees of freedom')
                pix_list, dist = match_lines1_pix(np.array(s), ref_pix_list)

            pix_list[dist > 1] = -1
            pixpos[msk] = pix_list
            
        identified_lines_tgt.append_order_info(o, wvl, indices, pixpos)

    # REF_TYPE = "OH"
    # fn = "%s_IGRINS_identified_%s_%s.json" % (REF_TYPE, band, helper.utdate)
    # item_path = caldb.query_item_path((band, master_obsid),
    #                                   "IDENTIFIED_LINES")
    # item_path = caldb.query_item_path((band, master_obsid),
    #                                   "IDENTIFIED_LINES")
    obsset.store(DESCS["IDENTIFIED_LINES_JSON"],
                 identified_lines_tgt.data)

def find_affine_transform(obsset):

    # As register.db has not been written yet, we cannot use
    # obsset.get("orders")
    orders = obsset.load(DESCS["ORDERS_JSON"])["orders"]

    ap = get_simple_aperture_from_obsset(obsset, orders)

    lines_data = obsset.load(DESCS["IDENTIFIED_LINES_JSON"])

    from .identified_lines import IdentifiedLines
    identified_lines_tgt = IdentifiedLines.load(lines_data)

    xy_list_tgt = identified_lines_tgt.get_xy_list_from_pixlist(ap)

    from .echellogram import Echellogram

    echellogram_data = obsset.rs.load_ref_data(kind="ECHELLOGRAM_JSON")

    echellogram = Echellogram.from_dict(echellogram_data)

    xy_list_ref = identified_lines_tgt.get_xy_list_from_wvllist(echellogram)

    assert len(xy_list_tgt) == len(xy_list_ref)

    from .fit_affine import fit_affine_clip
    affine_tr, mm = fit_affine_clip(np.array(xy_list_ref),
                                    np.array(xy_list_tgt))

    d = dict(xy1f=xy_list_ref, xy2f=xy_list_tgt,
             affine_tr_matrix=affine_tr.get_matrix(),
             affine_tr_mask=mm)

    obsset.store(DESCS["ALIGNING_MATRIX_JSON"],
                 data=d)


def _get_wavelength_solutions(affine_tr_matrix, zdata,
                              new_orders, nx=4096):
    """
    new_orders : output orders

    convert (x, y) of zdata (where x, y are pixel positions and z
    is wavelength) with affine transform, then derive a new wavelength
    solution.

    """
    #TODO: Input limited domain range for wavelength for RIMAS. Might not need to
    from .ecfit import get_ordered_line_data, fit_2dspec

    affine_tr = matplotlib.transforms.Affine2D()
    affine_tr.set_matrix(affine_tr_matrix)

    do_print = False
    d_x_wvl = {}
    for order, z in zdata.items():
        xy_T = affine_tr.transform(np.array([z.x, z.y]).T)
        if do_print:
            print("X:", z.x[953:955])
            print("Y:", z.y[953:955])
            print("WVL:", z.wvl[953:955])
            print("XT:", xy_T[953:955, 0])
            do_print = False
            tmp0 = z.wvl
            tmp1 = xy_T[:, 0]

        x_T = xy_T[:, 0]
        d_x_wvl[order] = (x_T, z.wvl)

    _xl, _ol, _wl = get_ordered_line_data(d_x_wvl)

    #idx = np.logical_and(_xl > 400, _xl < 1600)
    idx = np.logical_and(_xl > int(0.2*nx), _xl < int(0.8*nx))
    _xl2 = _xl[idx]
    _ol2 = _ol[idx]
    _wl2 = _wl[idx]

    # _xl : pixel
    # _ol : order
    # _wl : wvl * order

    x_domain = [0, nx-1]
    # orders = igrins_orders[band]
    # y_domain = [orders_band[0]-2, orders_band[-1]+2]
    y_domain = [new_orders[0], new_orders[-1]]
    p, m = fit_2dspec(_xl2, _ol2, _wl2, x_degree=4, y_degree=3,
                      x_domain=x_domain, y_domain=y_domain)

    xx = np.arange(nx)
    wvl_sol = []
    for o in new_orders:
        oo = np.empty_like(xx)
        oo.fill(o)
        wvl = p(xx, oo) / o
        wvl_sol.append(list(wvl))

    return wvl_sol


def transform_wavelength_solutions(obsset):

    # load affine transform

    # As register.db has not been written yet, we cannot use
    # obsset.get("orders")

    helper = ResourceHelper(obsset)
    orders = helper.get("orders")

    d = obsset.load(DESCS["ALIGNING_MATRIX_JSON"])

    affine_tr_matrix = d["affine_tr_matrix"]

    # load echellogram
    echellogram_data = obsset.rs.load_ref_data(kind="ECHELLOGRAM_JSON")

    from .echellogram import Echellogram
    echellogram = Echellogram.from_dict(echellogram_data)

    wvl_sol = _get_wavelength_solutions(affine_tr_matrix,
                                        echellogram.zdata,
                                        orders, nx=obsset.detector.nx)
    #print("REPLACING V0 WITH ECHELLOGRAM DATA")
    #for i in range(15):
    #    i2 = i+7
    #    wvl_sol[i2] = echellogram_data['wvl_list'][i]

    #TODO: REMOVE
    '''
    #TEST WAVELENGTH
    import matplotlib.pyplot as plt
    #plt.show()
    plt.figure('Wavelength Solution')
    plt.plot(wvl_sol[4+0], 'b', label='Wvl Sol')
    plt.plot(echellogram_data['wvl_list'][0], 'g', label='Echellogram')
    for i in range(1, 15):
        plt.plot(wvl_sol[4+i], 'b')
        plt.plot(echellogram_data['wvl_list'][i], 'g')
    plt.xlabel('X Index')
    plt.ylabel('Frequency')
    plt.legend(loc=0, prop={'size': 12})


    plt.figure('Wavelength Diff')
    for i in range(15):
        diff = wvl_sol[4+i] - np.array(echellogram_data['wvl_list'][i])
        denom = np.array(wvl_sol[4+i][1:]) - np.array(wvl_sol[4+i][:-1])
        #plt.plot(diff/wvl_sol[4+i]*100, label=str(i))
        plt.plot(diff[:-1]/denom*100, label=str(i))
    plt.xlabel('X Index')
    plt.ylabel('% Diff')
    plt.legend(loc=0, prop={'size': 12})
    #plt.show()
    '''

    obsset.store(DESCS["WVLSOL_V0_JSON"],
                 data=dict(orders=orders, wvl_sol=wvl_sol))

    return wvl_sol
# from ..libs.transform_wvlsol import transform_wavelength_solutions


def _make_order_flat(flat_normed, flat_mask, orders, order_map):

    # from storage_descriptions import (FLAT_NORMED_DESC,
    #                                   FLAT_MASK_DESC)

    # flat_normed  = flaton_products[FLAT_NORMED_DESC][0].data
    # flat_mask = flaton_products[FLAT_MASK_DESC].data

    import scipy.ndimage as ni
    slices = ni.find_objects(order_map)

    ny, nx = flat_normed.shape

    mean_order_specs = []
    mask_list = []
    for o in orders:
        # if slices[o-1] is None:
        #     continue
        sl = (slices[o-1][0], slice(0, nx))
        d_sl = flat_normed[sl].copy()
        d_sl[order_map[sl] != o] = np.nan

        f_sl = flat_mask[sl].copy()
        f_sl[order_map[sl] != o] = np.nan
        ff = np.nanmean(f_sl, axis=0)
        mask_list.append(ff)

        mmm = order_map[sl] == o

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'Mean of empty slice')

            ss = [np.nanmean(d_sl[2:-2][:, i][mmm[:, i][2: -2]])
                  for i in range(nx)]

        mean_order_specs.append(ss)

    from .trace_flat import (get_smoothed_order_spec,
                             get_order_boundary_indices)

    s_list = [get_smoothed_order_spec(s) for s in mean_order_specs]

    i1i2_list = [get_order_boundary_indices(s, s0, nx=nx)
                 for s, s0 in zip(mean_order_specs, s_list)]
    
    from .smooth_continuum import get_smooth_continuum
    s2_list = [get_smooth_continuum(s) for s, (i1, i2)
               in zip(s_list, i1i2_list)]
    #import matplotlib.pyplot as plt
    #plt.plot(s_list[0], 'b')
    #plt.plot(s2_list[0], 'r')
    #plt.plot(mean_order_specs[0], 'g')
    #plt.show()

    # make flat
    flat_im = np.ones(flat_normed.shape, "d")

    fitted_responses = []

    for o, px in zip(orders, s2_list):
        sl = (slices[o-1][0], slice(0, nx))
        d_sl = flat_normed[sl].copy()
        msk = (order_map[sl] == o)
        # d_sl[~msk] = np.nan

        d_div = d_sl / px
        px2d = px * np.ones_like(d_div)  # better way to broadcast px?
        with np.errstate(invalid="ignore"):
            d_div[px2d < 0.05*px.max()] = 1.

        flat_im[sl][msk] = (d_sl / px)[msk]
        fitted_responses.append(px)

    with np.errstate(invalid="ignore"):
        flat_im[flat_im < 0.5] = np.nan
    
    order_flat_dict = dict(orders=orders,
                           fitted_responses=fitted_responses,
                           i1i2_list=i1i2_list,
                           mean_order_specs=mean_order_specs)

    return flat_im, order_flat_dict


def save_orderflat(obsset):

    helper = ResourceHelper(obsset)
    orders = helper.get("orders")

    from .aperture_helper import get_simple_aperture_from_obsset

    ap = get_simple_aperture_from_obsset(obsset, orders=orders)

    order_map = ap.make_order_map()

    hdul = obsset.load_resource_for("flat_normed")
    flat_normed = get_first_science_hdu(hdul).data

    flat_mask = obsset.load_resource_for("flat_mask")

    order_flat_im, order_flat_json = _make_order_flat(flat_normed,
                                                      flat_mask,
                                                      orders, order_map)

    hdul = obsset.get_hdul_to_write(([], order_flat_im))
    obsset.store(DESCS["order_flat_im"], hdul)

    obsset.store(DESCS["order_flat_json"], order_flat_json)

    order_map2 = ap.make_order_map(mask_top_bottom=True)
    bias_mask = flat_mask & (order_map2 > 0)

    obsset.store(DESCS["bias_mask"], bias_mask)


def update_db(obsset):

    # save db
    obsset.add_to_db("register")


# steps = [Step("Make Combined Sky", make_combined_image_sky),
#          Step("Extract Simple 1d Spectra", extract_spectra),
#          Step("Identify Orders", identify_orders),
#          Step("Identify Lines", identify_lines),
#          Step("Find Affine Transform", find_affine_transform),
#          Step("Derive transformed Wvl. Solution", transform_wavelength_solutions),
#          Step("Save Order-Flats, etc", save_orderflat),
#          Step("Update DB", update_db),
# ]

