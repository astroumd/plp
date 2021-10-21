import os
import pickle

import numpy as np

igrins_orders = {}
igrins_orders["H"] = range(99, 122)
igrins_orders["K"] = range(72, 94)


def get_ordered_line_data(identified_lines, orders=None):
    """
    identified_lines : dict of lines with key of orders_i.
    lines[0] : list of x positions
    lines[1] : list of wavelengths
    """
    x_list, y_list, z_list = [], [], []
    # x:pixel, y:order, z:wavelength

    if orders is None:
        o_l = [(i, oh)  for i, oh in identified_lines.items()]
    else:
        o_l = zip(orders, identified_lines)

    for o, oh in sorted(o_l):
        x_list.extend(oh[0])
        y_list.extend([o] * len(oh[0]))
        z_list.extend(np.array(oh[1])*o)

    return map(np.array, [x_list, y_list, z_list])


def check_dx1(ax, x, y, dx, gi, mystd):

    grid_z2 = gi(x, y, dx)
    im = ax.imshow(grid_z2, origin="lower", aspect="auto",
                   extent=(gi.xi[0], gi.xi[-1], gi.yi[0], gi.yi[-1]),
                   interpolation="none")
    im.set_clim(-mystd, mystd)

def check_dx2(ax, x, y, dx):
    m1 = dx >= 0
    ax.scatter(x[m1], y[m1], dx[m1]*10, color="r")
    m2 = dx < 0
    ax.scatter(x[m2], y[m2], -dx[m2]*10, color="b")

class GridInterpolator(object):
    def __init__(self, xi, yi, interpolator="mlab"):
        self.xi = xi
        self.yi = yi
        self.xx, self.yy = np.meshgrid(xi, yi)
        self._interpolator = interpolator


    def _grid_scipy(self, xl, yl, zl):
        from scipy.interpolate import griddata
        x_sample = 256
        z_gridded = griddata(np.array([yl*x_sample, xl]).T,
                             np.array(zl),
                             (self.yy*x_sample, self.xx),
                             method="linear")
        return z_gridded


    def __call__(self, xl, yl, zl):
        if self._interpolator == "scipy":
            z_gridded = self._grid_scipy(xl, yl, zl)
        elif self._interpolator == "mlab":
            from matplotlib.mlab import griddata
            try:
                z_gridded = griddata(xl, yl, zl, self.xi, self.yi)
            except Exception:
                z_gridded = self._grid_scipy(xl, yl, zl)

        return z_gridded

def show_grided_image(ax, gi, xl, yl, zl, nx, orders):
    import matplotlib

    extent = [0, nx, orders[0]-1, orders[-1]+1]

    z_max, z_min = zl.max(), zl.min()
    norm = matplotlib.colors.Normalize(vmin=z_min, vmax=z_max)

    z_gridded = gi(xl, yl, zl)

    ax.imshow(z_gridded, aspect="auto", origin="lower", interpolation="none",
              extent=extent, norm=norm)

    ax.scatter(xl, yl, 10, c=zl, norm=norm)
    ax.set_xlim(0, nx)
    ax.set_ylim(orders[0]-1, orders[-1]+1)

def fit_2dspec(xl, yl, zl, x_degree=4, y_degree=3,
               x_domain=None, y_domain=None, p_init=None):
    from astropy.modeling import fitting
    # Fit the data using astropy.modeling
    if x_domain is None:
        x_domain = [min(xl), max(xl)]
    # more room for y_domain??
    if y_domain is None:
        #y_domain = [orders[0]-2, orders[-1]+2]
        y_domain = [min(yl), max(yl)]
    from astropy.modeling.polynomial import Chebyshev2D
    if p_init is None:
        p_init = Chebyshev2D(x_degree=x_degree, y_degree=y_degree,
                             x_domain=x_domain, y_domain=y_domain)
    f = fitting.LinearLSQFitter()

    p = f(p_init, xl, yl, zl)

    for i in [0]:
        dd = p(xl, yl) - zl
        m = np.abs(dd) < 3.*dd.std()
        p = f(p, xl[m], yl[m], zl[m])

    return p, m


def get_dx(xl, yl, zl, orders, p, nx):
    dlambda_order = {}
    for o in orders:
        wvl_minmax = p([0, nx-1], [o]*2) / o
        dlambda = (wvl_minmax[1] - wvl_minmax[0]) / nx
        dlambda_order[o] = dlambda

    dlambda = [dlambda_order[y1] for y1 in yl]
    dx = (zl - p(xl, yl))/yl/dlambda

    return dx


def get_dx_from_identified_lines(p, identified_lines, nx):
    dpix_list = {}
    for i, oh in sorted(identified_lines.items()):
        oh = identified_lines[i]
        o = i #orders[i]
        wvl = p(oh[0], [o]*len(oh[0])) / o

        wvl_minmax = p([0, nx-1], [o]*2) / o
        dlambda = (wvl_minmax[1] - wvl_minmax[0]) / nx

        dpix_list[i] = (oh[1] - wvl)/dlambda

    return dpix_list

