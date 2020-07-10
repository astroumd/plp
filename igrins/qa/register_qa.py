import numpy as np
import itertools
from matplotlib.offsetbox import AnchoredText

def produce_qa(obsset, outtype="pdf"):
    
    from matplotlib.figure import Figure
    from ..quicklook.qa_helper import save_figlist, check_outtype
    from ..igrins_libs.path_info import get_zeropadded_groupname

    order_flat_json = obsset.load("order_flat_json")
    
    fig_list = check_order_flat(order_flat_json)

    from igrins import DESCS

    section, _outroot = DESCS["QA_ORDERFLAT_DIR"], "qa_orderflat"
    obsdate, band = obsset.get_resource_spec()
    groupname = get_zeropadded_groupname(obsset.groupname)
    outroot = "SDC{}_{}_{}_{}".format(band, obsdate, groupname, _outroot)
    #print("SSS:", section, outroot, outtype)
    #zzz
    save_figlist(obsset, fig_list, section, outroot, outtype)

def check_order_flat(order_flat_json):

    from ..procedures.trace_flat import (prepare_order_trace_plot,
                                         check_order_trace1, check_order_trace2)

    # from storage_descriptions import ORDER_FLAT_JSON_DESC

    mean_order_specs = order_flat_json["mean_order_specs"]

    from ..procedures.trace_flat import (get_smoothed_order_spec,
                                         get_order_boundary_indices,
                                         get_order_flat1d)

    # these are duplicated from make_order_flat
    s_list = [get_smoothed_order_spec(s) for s in mean_order_specs]
    i1i2_list = [get_order_boundary_indices(s, s0) \
                 for s, s0 in zip(mean_order_specs, s_list)]
    # p_list = [get_order_flat1d(s, i1, i2) for s, (i1, i2) \
    #           in zip(s_list, i1i2_list)]

    from ..procedures.smooth_continuum import get_smooth_continuum
    s2_list = [get_smooth_continuum(s) for s, (i1, i2) \
               in zip(s_list, i1i2_list)]

    fig_list, ax_list = prepare_order_trace_plot(s_list)
    x = np.arange(2048)
    for s, i1i2, ax in zip(mean_order_specs, i1i2_list, ax_list):
        check_order_trace1(ax, x, s, i1i2)

    for s, s2, ax in zip(mean_order_specs, s2_list, ax_list):
        ax.plot(x, s2)
        #check_order_trace2(ax, x, p)

    return fig_list

