import pandas as pd
import numpy as np


def _convert2wvlsol(p, orders_w_solutions, nx=2048):

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

def _fit_2d(xl, yl, zlo, xdeg=4, ydeg=3, p_init=None, nx=2048):
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


def fit_wvlsol(df, xdeg=4, ydeg=3, p_init=None, nx=2048):
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

    p, fit_results = _fit_2d(xl, yl, zlo, xdeg=xdeg, ydeg=ydeg, p_init=p_init, nx=nx)
    return p, fit_results


def derive_wvlsol(obsset):

    d = obsset.load("SKY_FITTED_PIXELS_JSON")
    df = pd.DataFrame(**d)

    nx = obsset.detector.nx

    msk = df["slit_center"] == 0.5
    dfm = df[msk]

    p, fit_results = fit_wvlsol(dfm, nx=nx)

    from ..igrins_libs.resource_helper_igrins import ResourceHelper
    helper = ResourceHelper(obsset)
    orders = helper.get("orders")

    wvl_sol = _convert2wvlsol(p, orders, nx=obsset.detector.nx)
    d = dict(orders=orders,
             wvl_sol=wvl_sol)
    
    obsset.store("SKY_WVLSOL_JSON", d)

    fit_results["orders"] = orders
    obsset.store("SKY_WVLSOL_FIT_RESULT_JSON",
                 fit_results)


