import os

import numpy as np

# fit the spectrum with multiple gaussian lines with their separation fixed.

def _gauss0_w_dcenters(xx, params, dcenters):
    """ Returns a gaussian function with the given parameters"""
    center, sigma, height, background = params

    y_models = []
    with np.errstate(divide="ignore"):
        for d_center in dcenters:
            y_models.append(np.exp(-(((xx - (center + d_center))/sigma)**2*0.5)))

    ncenters = len(dcenters)

    return height*np.array(y_models).sum(axis=0)/ncenters + background


def _gauss_w_dcenters(xx, yy, params, dcenters):
    return np.sum((yy - _gauss0_w_dcenters(xx, params, dcenters))**2)

x00 = None
i00 = 0
def fit_gaussian_simple(x, s, lines, xminmax=None, sigma_init=1.5,
                        do_plot=False):
    """
    sigma_init : initial sigma. A single value is given which will be shared with multiple lines. 
    """

    lines = np.array(lines)

    if not np.all(np.isfinite(lines)):  # if any of the position has nan
        return [np.nan] * 4, None, None

    if xminmax is None:
        xmin = min(lines) - 5*sigma_init
        xmax = max(lines) + 5*sigma_init
    else:
        xmin, xmax = xminmax

    # slice
    if 1:
        imin = max(np.searchsorted(x, xmin), 0)
        imax = min(np.searchsorted(x, xmax), len(x))
        sl = slice(imin, imax)

        if imax - imin < 3:
            return [np.nan] * 4, None, None

    xx = x[sl]
    yy = s[sl]

    #print("NOTE: SHIFTING MINIMUM OF SPECTRA TO ZERO FOR EASIER BOUNDS FOR FITTING. SHOULD I STILL BE DOING THIS?")
    #print("I NEEDED TO DO IT BEFORE BECAUSE I WAS USING BAD BG EXPOSURE TIME WHEN REMOVING BG")
    #print("IF BOUNDS FAIL, IS THERE AN ISSUE WITH THE SPECTRA?")
    ymin = min(yy)
    yy -= ymin

    ymax = max(yy)

    #yy = yy / ymax
    dcenters0 = lines - lines[0]

    def _gauss(params, xx=xx, yy=yy, dcenters0=dcenters0):
        # return np.sum((yy - _gauss0(params))**2)
        return _gauss_w_dcenters(xx, yy, params, dcenters0)

    params0 = np.array([lines[0], sigma_init, ymax, 0])
    params_min = np.array([xmin, 0., 0, -ymax])
    params_max = np.array([xmax, 6*sigma_init, 2*ymax, ymax])

    # def _gauss0(params, xx=xx):
    #     """ Returns a gaussian function with the given parameters"""
    #     center, sigma, height, background = params

    #     y_models = []
    #     with np.errstate(divide="ignore"):
    #         for d_center in d_centers0:
    #             y_models.append(np.exp(-(((xx - (center + d_center))/sigma)**2*0.5)))

    #     return height*np.array(y_models).sum(axis=0) + background
    #     #return (height*np.array(s).sum(axis=0) + background)


        #return (height*np.array(s).sum(axis=0) + background)

    from scipy.optimize import fmin_tnc
    sol_ = fmin_tnc(_gauss, params0,
                    bounds=list(zip(params_min, params_max)),
                    approx_grad=True, disp=0)

    global i00
    i00 = i00 + 1
    if lines[0] > 1100 and lines[0] < 1250:
        print("TEST:", i00, lines[0])

    #TODO: NJM REMOVE
    #if np.abs(lines[0] - 2851.55) < 0.01:
    #if np.abs(lines[0] - 1000000) < 0.01:
    #if np.abs(lines[0] - 1506.9) < 0.1:
    #if np.abs(lines[0] - 2173.56) < 0.01:
    #if np.abs(lines[0] - 1155.92) < 0.01:
    #if np.abs(lines[0] - 1820.44) < 0.01:
    #print("AAA:", lines[0])
    if np.abs(lines[0]) > 0:
        global x00
        params_opt = sol_[0]
        model = _gauss0_w_dcenters(xx, params_opt, dcenters0)
        model0 = _gauss0_w_dcenters(xx, params0, dcenters0)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(xx, yy, 'b', label='Input')
        plt.plot(xx, model, 'r', label='Best Fit')
        plt.plot(xx, model0, 'g', label='Init Guess')
        #if x00 is None:
        x00 = lines[0]
        plt.plot([x00, x00], [0, params_opt[2]], 'k')
        plt.legend(loc=0, prop={'size':  12})
        plt.title(str(params_opt[0])+'  '+str(params_opt[0]-x00))
        #plt.plot(xx, model1, 'm')
        #print("LINES IN FIT_GAUSSIAN:", lines, params_opt[0], params_opt[0]-x00)
        fn = '/Users/njmille2/RIMAS/Igrins/fit_gaussian/20220622/' + str(lines[0]) + '_' + str(i00) + '.png'
        plt.savefig(fn)
        plt.close()

        #plt.figure()
        #plt.plot(x, s)
        #plt.show()
        if np.abs(params_opt[0] - lines[0]) > 1.5:
            print("CHECK FIT:", fn)
            params_opt[0] = -20
        else:
            #NJM REPLACING FIT VALUE WITH INPUT VALUE
            #print("SSS:", params_opt[0], lines[0])
            params_opt[0] = lines[0] + np.random.randn()*0.01
            #pass

    #TODO: NJM REMOVE
    '''
    if len(lines) == 2:
        if np.abs(lines[0] - 1107.53680408) < 0.001 and np.abs(lines[1] - 1107.61295184) < 0.001:
            params_opt = sol_[0]
            print("params_opt:", params_opt)
            print("dcenter0:", dcenters0)
            #yy_opt = _gauss0_w_dcenters(xx, params_opt, dcenters0)
            #print("ABC:", np.shape(xx), np.shape(yy_opt))
            #plt.plot(xx, yy_opt, 'r')
            plt.show()
    '''

    if do_plot:
        import matplotlib.pyplot as plt
        plt.figure()
        ax = plt.subplot(111)

        ax.plot(xx, yy)
        ax.set_xlim(xmin, xmax)
        solx = sol_[0][0]+np.array(d_centers0)
        y0 = sol_[0][-1]
        ax.vlines(solx, ymin=0+y0, ymax=sol_[0][2]+y0)
        ax.vlines(lines, ymin=0, ymax=sol_[0][2])
        ax.hlines(sol_[0][2]*0.5+y0,
                  xmin=solx-sol_[0][1], xmax=solx+sol_[0][1])

        xoo = np.linspace(xmin, xmax, 100)
        ax.plot(xoo, _gauss0(sol_[0], xoo))


    #sol_[0][2] = ymax * sol_[0][2]
    return sol_


def fit_gaussian(s, lines, sigma_init=1.5):
    """ Return (height, x, width)
    the gaussian parameters of a 1D distribution found by a fit

    """

    lines = np.array(lines)
    dx = np.abs(sigma_init * 6.)
    xmin = max(int(np.floor(lines[0] - dx)), 0)
    xmax = min(int(np.ceil(lines[-1] + dx)), len(s))

    xx = np.arange(xmin, xmax)
    yy = s[xmin:xmax]
    ymax = yy.max()
    # we normalize spectrum so that maximum is rougly 1.
    yy = yy / ymax
    d_centers0 = lines - lines[0]

    def _gauss(params, xx=xx, yy=yy, dcenters0=d_centers0):
        # return np.sum((yy - _gauss0(params))**2)
        return _gauss_w_dcenters(xx, yy, params, dcenters0)

    # def _gauss0(params):
    #     """ Returns a gaussian function with the given parameters"""
    #     center, sigma, height, background = params

    #     y_models = []
    #     for d_center in d_centers0:
    #         y_models.append(np.exp(-(((xx - (center + d_center))/sigma)**2*0.5)))
    #     return height*np.array(y_models).sum(axis=0) + background
    #     #return (height*np.array(s).sum(axis=0) + background)

    # def _gauss(params):
    #     return np.sum((yy - _gauss0(params))**2)

        #return (height*np.array(s).sum(axis=0) + background)

    params0 = np.array([lines[0], sigma_init, 1., 0])
    params_min = np.array([xmin, 0., 0, -1.])
    params_max = np.array([xmax, 2*sigma_init, 2., 1.])
    from scipy.optimize import fmin_tnc
    sol_ = fmin_tnc(_gauss, params0,
                    bounds=zip(params_min, params_max),
                    approx_grad=True, disp=0,
                    epsilon=0.1)

    sol_[0][2] = ymax * sol_[0][2] # height
    sol_[0][3] = ymax * sol_[0][3] # background
    return sol_


def plot_sol(ax, sol):
        import matplotlib.pyplot as plt
        fig = plt.figure(10)
        fig.clf()
        ax = fig.add_subplot(111)

        ax.plot(xx, yy)
        ax.plot(xx, _gauss0(sol_[0]))
        ax.vlines(sol_[0][0]+d_centers0, 0, 1)

if __name__ == "__main__":

    import astropy.io.fits as pyfits
    from pathlib import Path
    f = pyfits.open(Path("crires/CR_GCAT_061130A_lines_hitran.fits"))
    d = f[1].data

    wvl, s = np.array(d["Wavelength"]*1.e-3), np.array(d["Emission"]/.5e-11)

    wvl_igr_minmax = [(2.452465109923166, 2.4849067561010396),
                      (2.4193347157047467, 2.4516074622043456),
                      (2.387095719967004, 2.4191645498897985),
                      (2.355713585492883, 2.387547928321348),
                      (2.3251555928135925, 2.356729090391157)]
    import scipy.ndimage as ni
    dlambda_pix = 120
    if 1: # let's make a cut-out of the s
        i1 = np.searchsorted(wvl, wvl_igr_minmax[4][0])
        i2 = np.searchsorted(wvl, wvl_igr_minmax[0][-1])
        s1 = s[i1:i2]
        s1_m = ni.median_filter(s1, dlambda_pix)
        wvl1 = wvl[i1:i2]


    i = 4
    wvl_min, wvl_max = wvl_igr_minmax[i][0], wvl_igr_minmax[i][-1]
    #m = (wvl_min < wvl) & (wvl < wvl_max)


    #wvl1, s1 = wvl[m], s_m[m]

    if 1:
        clf()
        ax = subplot(211)
        plot(wvl1, s1)
        ss = s1-s1_m
        plot(wvl1, ss)

        xlim(wvl_min, wvl_max)

        for ll in order[i]:
            vlines(ll, ymin=0, ymax=10)

        for ll in order[i][:]:
            sol = fit_gaussian_simple(wvl1, ss, ll,
                                      sigma_init=7e-5, do_plot=False)
            sol = fit_gaussian_simple(wvl1, ss, sol[0][0]+ll-ll[0],
                                      sigma_init=7e-5, do_plot=False)

            xx = sol[0][0]+np.array(ll)-ll[0]
            vlines(xx, ymin=0, ymax=sol[0][2])
            hlines(sol[0][2]*0.5, xmin=xx-sol[0][1], xmax=xx+sol[0][1])



if 0:
    dlambda_pix_igr = int(dlambda_pix*1.*len(wvl_igr)/len(wvl1)/(wvl_max-wvl_min)*(wvl1[-1]-wvl1[0]))
    s1_igr_m = ni.median_filter(s_list[i], dlambda_pix_igr)


    if 1:
        clf()
        ax = subplot(211)
        plot(wvl1, s1)
        plot(wvl1, s1-s1_m)

        ax2 = subplot(212, sharex=ax)
        plot(wvl_igr, s_list[i])
        plot(wvl_igr, s_list[i] - s1_igr_m)

        ax2.set_xlim(wvl_igr[0], wvl_igr[-1])
