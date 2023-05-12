import pandas as pd
import numpy as np


def _convert2wvlsol(p, orders_w_solutions, nx):

    # derive wavelengths.
    xx = np.arange(nx)
    wvl_sol = []
    for o in orders_w_solutions:
        oo = np.empty_like(xx)
        oo.fill(o)
        wvl = p(xx, oo) / o
        wvl_sol.append(list(wvl))

    return wvl_sol


from .ecfit import fit_2dspec

def _fit_2d(xl, yl, zlo, nx, xdeg=4, ydeg=3, p_init=None):
    """
    df: pixels, order, wavelength"""

    x_domain = [0, nx-1]
    y_domain = [min(yl)-2, max(yl)+2]

    msk = np.isfinite(xl)

    fit_params = dict(x_degree=xdeg, y_degree=ydeg,
                      x_domain=x_domain, y_domain=y_domain)

    p, m = fit_2dspec(xl[msk], yl[msk], zlo[msk], p_init=p_init, **fit_params)

    from .astropy_poly_helper import serialize_poly_model
    poly_2d = serialize_poly_model(p)
    fit_results = dict(xyz=[xl[msk], yl[msk], zlo[msk]],
                       fit_params=fit_params,
                       fitted_model=poly_2d,
                       fitted_mask=m)

    return p, fit_results


def fit_wvlsol(df, nx, xdeg=4, ydeg=3, p_init=None):
    """
    df: pixels, order, wavelength"""
    from .ecfit import fit_2dspec

    xl = df["pixels"].values
    yl = df["order"].values
    zl = df["wavelength"].values
    zlo = zl * yl
    # xl : pixel
    # yl : order
    # zlo : wvl * order

    p, fit_results = _fit_2d(xl, yl, zlo, nx, xdeg=xdeg, ydeg=ydeg, p_init=p_init)
    return p, fit_results


def derive_wvlsol(obsset, fit_type='sky'):

    if fit_type == 'sky':
        d = obsset.load("SKY_FITTED_PIXELS_JSON")
    else:
        d = obsset.load("THAR_FITTED_PIXELS_JSON")
    df = pd.DataFrame(**d)

    nx = obsset.detector.nx

    msk = df["slit_center"] == 0.5
    dfm = df[msk]

    #print("REMOVING BAD GAUSSIAN FITS (-20) FROM WVL SOLUTION FIT")
    msk = np.ones(len(dfm), dtype=bool)
    params = dfm['params']
    for i in range(len(dfm)):
        if params[i][0] == -20:
            print(i, params[i])
            msk[i] = False
    dfm = dfm[msk]

    p, fit_results = fit_wvlsol(dfm, nx)

    from ..igrins_libs.resource_helper_igrins import ResourceHelper
    helper = ResourceHelper(obsset)
    orders = helper.get("orders")
    
    #NJM Added xrange to constrain spectra ranges based on range of fits for wavelength
    #Added to both outputs, but should remove from one that we don't want
    xrange = [np.min(dfm['pixels']), np.max(dfm['pixels'])]

    wvl_sol = _convert2wvlsol(p, orders, nx)
    d = dict(orders=orders,
             wvl_sol=wvl_sol,
             xrange=xrange)
   
    obsset.store("SKY_WVLSOL_JSON", d)


    fit_results["orders"] = orders

    #fit_results["xrange"] = xrange
    
    obsset.store("SKY_WVLSOL_FIT_RESULT_JSON",
                 fit_results)


