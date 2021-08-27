import os

import numpy as np
import pandas as pd

from ..pipeline.steps import Step, ArghFactoryWithShort
from ..igrins_libs.logger import logger

from ..igrins_libs.a0v_obsid import get_group2, get_a0v_obsid
from ..igrins_libs.igrins_config import IGRINSConfig
from ..igrins_libs.oned_spec_helper import OnedSpecHelper
from ..igrins_libs.path_info import ensure_dir, get_zeropadded_groupname
from ..igrins_libs.resource_helper_igrins import ResourceHelper


def set_basename_postfix(obsset, basename_postfix):
    obsset.set_basename_postfix(basename_postfix=basename_postfix)


def _plot_source_spec(fig, tgt, objname=""):

    ax1a = fig.add_subplot(211)
    ax1b = fig.add_subplot(212, sharex=ax1a)

    for wvl, s, sn, dn in zip(tgt.um,
                              tgt.spec, tgt.sn, tgt.domain):
        #s[s<0] = np.nan
        #sn[sn<0] = np.nan

        npts = dn[1] - dn[0] + 1
        wvl = wvl[dn[0]:dn[1]+1]
        s = s[:npts]
        sn = sn[:npts]

        #s = s[7:-7]
        #sn = sn[7:-7]
        #wvl = wvl[7:-7]
        
        ax1a.plot(wvl, s)
        ax1b.plot(wvl, sn)

    ax1a.set_ylabel("Counts [DN]")
    ax1b.set_ylabel("S/N per Res. Element")
    ax1b.set_xlabel("Wavelength [um]")

    if objname:
        ax1a.set_title(objname)

def get_tgt_spec_cor(obsset, tgt, a0v, threshold_a0v, multiply_model_a0v):
    tgt_spec_cor = []
    #for s, t in zip(s_list, telluric_cor):
    for s, t, t2, dn in zip(tgt.spec,
                            a0v.spec,
                            a0v.flattened,
                            tgt.domain):

        npts = dn[1] - dn[0] + 1
        s = s[:npts]
        t = t[:npts]
        t2 = t2[:npts]

        st = s/t
        msk = np.isfinite(t)
        if np.any(msk):
            #print np.percentile(t[np.isfinite(t)], 95), threshold_a0v
            t0 = np.percentile(t[msk], 95)*threshold_a0v
            st[t<t0] = np.nan

            st[t2 < threshold_a0v] = np.nan

        tgt_spec_cor.append(st)


    if multiply_model_a0v:
        # multiply by A0V model
        d = obsset.rs.load_ref_data("VEGA_SPEC")
        from ..procedures.a0v_spec import A0VSpec
        a0v_model = A0VSpec(d)

        a0v_interp1d = a0v_model.get_flux_interp1d(1.3, 2.5,
                                                   flatten=True,
                                                   smooth_pixel=32)
        for wvl, s in zip(tgt.um,
                          tgt_spec_cor):

            aa = a0v_interp1d(wvl)
            s *= aa


    return tgt_spec_cor


def _plot_div_a0v_spec(fig, tgt, obsset, a0v="GROUP2", a0v_obsid=None,
                       threshold_a0v=0.1,
                       objname="",
                       multiply_model_a0v=False,
                       html_output=False,
                       a0v_basename_postfix=""):
    # FIXME: This is simple copy from old version.

    a0v_obsid = get_a0v_obsid(obsset, a0v, a0v_obsid)
    if a0v_obsid is None:
        a0v_obsid_ = obsset.query_resource_basename("a0v")
        a0v_obsid = obsset.rs.parse_basename(a0v_obsid_)

    logger.warn("using A0V:{}".format(a0v_obsid))

    a0v_obsset = type(obsset)(obsset.rs, "A0V_AB", [a0v_obsid], ["A"],
                              basename_postfix=a0v_basename_postfix)

    a0v = OnedSpecHelper(a0v_obsset)

    # if True:

    #     if (a0v_obsid is None) or (a0v_obsid == "1"):
    #         A0V_basename = extractor.basenames["a0v"]
    #     else:
    #         A0V_basename = "SDC%s_%s_%04d" % (band, utdate, int(a0v_obsid))
    #         print(A0V_basename)

    #     a0v = extractor.get_oned_spec_helper(A0V_basename,
    #                                          basename_postfix=basename_postfix)
    # config = obsset.rs.config

    tgt_spec_cor = get_tgt_spec_cor(obsset, tgt, a0v,
                                    threshold_a0v,
                                    multiply_model_a0v)

    ax2a = fig.add_subplot(211)
    ax2b = fig.add_subplot(212, sharex=ax2a)

    #from ..libs.stddev_filter import window_stdev

    for wvl, s, t, dn in zip(tgt.um,
                             tgt_spec_cor,
                             a0v.flattened,
                             tgt.domain):

        npts = dn[1] - dn[0] + 1
        wvl = wvl[dn[0]:dn[1]+1]
        s = s[:npts]
        t = t[:npts]

        #wvl = wvl[7:-7]
        #s = s[7:-7]
        #t = t[7:-7]

        ax2a.plot(wvl, t, "0.8", zorder=0.5)
        ax2b.plot(wvl, s, zorder=0.5)

    s_max_list = []
    s_min_list = []
    for s in tgt_spec_cor[3:-3]:
        s_max_list.append(np.nanmax(s))
        s_min_list.append(np.nanmin(s))
    s_max = np.max(s_max_list)
    s_min = np.min(s_min_list)
    ds_pad = 0.05 * (s_max - s_min)

    ax2a.set_ylabel("A0V flattened")
    ax2a.set_ylim(-0.05, 1.1)
    ax2b.set_ylabel("Target / A0V")
    ax2b.set_xlabel("Wavelength [um]")

    ax2b.set_ylim(s_min-ds_pad, s_max+ds_pad)
    ax2a.set_title(objname)

def figlist_to_pngs(rootname, figlist, postfixes=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from itertools import count

    if postfixes is None:
        postfixes = ("fig%02d" % i for i in count(1))

    for postfix, fig in zip(postfixes, figlist):
        FigureCanvasAgg(fig)
        fig.savefig("%s_%s.png" % (rootname, postfix))
        #fig2.savefig("align_zemax_%s_fig2_fit.png" % postfix)
        #fig3.savefig("align_zemax_%s_fig3_hist_dlambda.png" % postfix)

def _save_to_pngs(fig_list, path, mastername):
    basename_postfix = None

    tgt_basename = mastername

    #dirname = "spec_" + tgt_basename
    #dirname = tgt_basename

    basename_postfix_s = basename_postfix if basename_postfix is not None else ""
    filename_prefix = "spec_" + tgt_basename + basename_postfix_s

    figout = os.path.join(path, filename_prefix)

    if not os.path.exists(path):
        os.mkdir(path)

    #Function is copied from old code
    figlist_to_pngs(figout, fig_list)

def get_fixed_i1i2_list(order_indices, i1i2_list):
    i1i2_list2 = []
    for o_index in order_indices:

        i1i2_list2.append(i1i2_list[o_index])
    return i1i2_list2

def get_i1i2_list(orders_w_solutions, new_orders, i1i2_list_):
    
    order_indices = []

    for o in orders_w_solutions:
        o_new_ind = np.searchsorted(new_orders, o)
        order_indices.append(o_new_ind)

    i1i2_list = get_fixed_i1i2_list(order_indices, i1i2_list_)

    return i1i2_list

def _save_to_html(config, groupname, tgt, utdate, band, orders_w_solutions,
                  i1i2_list, basename_postfix=None,
                  a0v=None, tgt_spec_cor=None):
    
    basename_postfix = None if basename_postfix == '' else basename_postfix
    if basename_postfix is not None:
        igr_log.warn("For now, no html output is generated if basename-postfix option is used")
    else:
        dirname = config.get_value('HTML_PATH', utdate)

        objroot = get_zeropadded_groupname(groupname)
        html_save(utdate, dirname, objroot, band,
                  orders_w_solutions, tgt.um,
                  tgt.spec, tgt.sn, i1i2_list, tgt.domain)

        if a0v is not None:
            objroot = get_zeropadded_groupname(groupname)+"A0V"

            html_save(utdate, dirname, objroot, band,
                      orders_w_solutions, tgt.um,
                      a0v.flattened, tgt_spec_cor, i1i2_list,
                      tgt.domain,
                      spec_js_name="jj_a0v.js")


def plot_spec(obsset, interactive=False,
              multiply_model_a0v=False,
              html_output=False,
              threshold_a0v=0.2):
    obsdate, band = obsset.get_resource_spec()

    config = IGRINSConfig(expt=obsset.expt)
    recipe = obsset.recipe_name
    target_type, nodding_type = recipe.split("_")
    master_obsid = obsset.master_obsid
    groupname = obsset.groupname

    helper = ResourceHelper(obsset)
    orders_w_solutions = helper.get("orders")

    a0v = None
    tgt_spec_cor = None

    if target_type in ["A0V"]:
        FIX_TELLURIC = False
    elif target_type in ["STELLAR", "EXTENDED"]:
        FIX_TELLURIC = True
    else:
        raise ValueError("Unknown recipe : %s" % recipe)

    tgt = OnedSpecHelper(obsset, basename_postfix=obsset.basename_postfix)

    do_interactive_figure = interactive

    if do_interactive_figure:
        from matplotlib.pyplot import figure as Figure
    else:
        from matplotlib.figure import Figure

    fig_list = []

    fig1 = Figure(figsize=(12, 6))
    fig_list.append(fig1)

    _plot_source_spec(fig1, tgt)

    if FIX_TELLURIC:
        fig1 = Figure(figsize=(12, 6))
        fig_list.append(fig1)

        _plot_div_a0v_spec(fig1, tgt, obsset,
                           multiply_model_a0v=multiply_model_a0v)
            
        if html_output:
   
            a0v_obsid = None
            a0v_obsid = get_a0v_obsid(obsset, a0v, a0v_obsid)
            if a0v_obsid is None:
                a0v_obsid_ = obsset.query_resource_basename("a0v")
                a0v_obsid = obsset.rs.parse_basename(a0v_obsid_)
            a0v_obsset = type(obsset)(obsset.rs, "A0V_AB", [a0v_obsid], ["A"],
                                      basename_postfix="")

            a0v = OnedSpecHelper(a0v_obsset)

            tgt_spec_cor = get_tgt_spec_cor(obsset, tgt, a0v,
                                            threshold_a0v,
                                            multiply_model_a0v)
            
            

    if fig_list:
        for fig in fig_list:
            fig.tight_layout()

        if not (do_interactive_figure or html_output):
            mastername = obsset.rs.basename_helper.to_basename(obsset.master_obsid)
            path = os.path.join(config.get_value("QA_PATH", obsdate),
                                obsset.recipe_name)
            _save_to_pngs(fig_list, path, mastername)
    
    if html_output:
        prod = obsset.load_resource_for("order_flat_json")

        new_orders = prod["orders"]
        i1i2_list_ = prod["i1i2_list"]

        i1i2_list = get_i1i2_list(orders_w_solutions, new_orders, i1i2_list_)
        
        _save_to_html(config, groupname, tgt, obsdate, band, orders_w_solutions,
                      i1i2_list, basename_postfix=obsset.basename_postfix,
                      a0v=a0v, tgt_spec_cor=tgt_spec_cor)

    if do_interactive_figure:
        import matplotlib.pyplot as plt
        plt.show()

def html_save(utdate, dirname, objroot, band,
              orders_w_solutions, wvl_solutions,
              tgt_spec, tgt_sn, i1i2_list, tgt_dn,
              spec_js_name="jj.js"):

        wvl_list_html, s_list_html, sn_list_html = [], [], []

        for wvl, s, sn, (i1, i2), dn in zip(wvl_solutions,
                                            tgt_spec, tgt_sn,
                                            i1i2_list, tgt_dn):

            #sl = slice(i1, i2)
            #
            #wvl_list_html.append(wvl[sl])
            #s_list_html.append(s[sl])
            #sn_list_html.append(sn[sl])

            #i1i2_list has 4, 4 for bad, extrapolated orders
            if i1 == 4 and i2 == 4:
                continue
           
            npts = dn[1] - dn[0] + 1

            wvl_tmp = wvl[dn[0]:dn[1]+1]
            wvl_tmp = wvl_tmp[3:-3]
            wvl_list_html.append(wvl_tmp)
            s_list_html.append(s[3:npts-3])
            sn_list_html.append(sn[3:npts-3])
            
            #wvl_list_html.append(wvl[dn[0]:dn[1]+1])
            #s_list_html.append(s[:npts])
            #sn_list_html.append(sn[:npts])

        save_for_html(dirname, objroot, band,
                      orders_w_solutions,
                      wvl_list_html, s_list_html, sn_list_html)

        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader('jinja_templates'))
        spec_template = env.get_template('spec.html')

        master_root = "igrins_spec_%s_%s" % (objroot, band)
        jsname = master_root + ".js"
        ss = spec_template.render(utdate=utdate,
                                  jsname=jsname,
                                  spec_js_name=spec_js_name)
        htmlname = master_root + ".html"
        open(os.path.join(dirname, htmlname), "w").write(ss)

def save_for_html(dir, name, band, orders, wvl_sol, s_list1, s_list2):

    ensure_dir(dir)

    # Pandas requires the byte order of data (from fits) needs to be
    # converted to native byte order of the computer this script is
    # running.
    wvl_sol = [w.byteswap().newbyteorder() for w in wvl_sol]
    s_list1 = [s1.byteswap().newbyteorder() for s1 in s_list1]
    s_list2 = [s2.byteswap().newbyteorder() for s2 in s_list2]

    df_even_odd = {}
    for o, wvl, s in zip(orders, wvl_sol, s_list1):
        if len(wvl) < 2: continue
        oo = ["even", "odd"][o % 2]
        dn = 'order_%s'%oo
        df = pd.DataFrame({dn: s},
                          index=wvl)
        df[dn][wvl[0]] = "NaN"
        df[dn][wvl[-1]] = "NaN"

        df_even_odd.setdefault(oo, []).append(df)

    df_list = [pd.concat(v).fillna("NaN") for v in df_even_odd.values()]
    df1 = df_list[0].join(df_list[1:], how="outer")

    #df_list = []
    df_even_odd = {}
    for o, wvl, s in zip(orders, wvl_sol, s_list2):
        if len(wvl) < 2: continue
        oo = ["even", "odd"][o % 2]
        dn = 'order_%s'%oo
        df = pd.DataFrame({dn: s},
                          index=wvl)

        df[dn][wvl[0]] = "NaN"
        df[dn][wvl[-1]] = "NaN"

        df_even_odd.setdefault(oo, []).append(df)

        #df_list.append(df)
    df_list = [pd.concat(v).fillna("NaN") for v in df_even_odd.values()]
    df2 = df_list[0].join(df_list[1:], how="outer")

    igrins_spec_output1 = "igrins_spec_%s_%s_fig1.csv.html" % (name, band)
    igrins_spec_output2 = "igrins_spec_%s_%s_fig2.csv.html" % (name, band)


    df1.to_csv(os.path.join(dir, igrins_spec_output1))
    df2.to_csv(os.path.join(dir, igrins_spec_output2))

    wvlminmax_list = []
    filtered_orders = []
    for o, wvl in zip(orders, wvl_sol):
        if len(wvl) > 2:
            filtered_orders.append(o)
            wvlminmax_list.append([min(wvl), max(wvl)])

    f = open(os.path.join(dir, "igrins_spec_%s_%s.js"%(name, band)),"w")
    f.write('name="%s : %s";\n' % (name,band))
    f.write("wvl_ranges=")
    f.write(str(wvlminmax_list))
    f.write(";\n")
    f.write("order_minmax=[%d,%d];\n" % (filtered_orders[0],
                                         filtered_orders[-1]))

    f.write('first_filename = "%s";\n' % igrins_spec_output1)
    f.write('second_filename = "%s";\n' % igrins_spec_output2)

    f.close()

steps = [Step("Set basename_postfix", set_basename_postfix,
              basename_postfix=''),
         Step("Plot spec", plot_spec,
              interactive=ArghFactoryWithShort(False),
              multiply_model_a0v=ArghFactoryWithShort(False),
              html_output=ArghFactoryWithShort(False))
]
