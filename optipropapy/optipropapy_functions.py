# general python libraries
import math
import csv

# external libraries
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from pathlib import Path

# animation libraries
from matplotlib.animation import FuncAnimation, PillowWriter


# randomly gen space objects desired for detection
def gen_sourcefield_random(n, sz, lu, min_pcnt):
    # create field coordinates
    x = np.linspace(-sz / 2, sz / 2, n + 1)

    # initialize source field array
    field_s = np.zeros((n + 1, n + 1))

    # create random arrays of luminosity and number of objects in each quadrant
    lumin = np.random.uniform(lu * min_pcnt, lu, size=(2, 2, 2))
    num_obj = np.random.randint(0, 3, size=(2, 2))

    # add objects to scene
    for ii in range(2):
        for jj in range(2):
            if num_obj[ii, jj] == 0:
                # do nothing
                continue
            elif num_obj[ii, jj] == 1:
                # create one object
                field_s[int((3**ii) * n / 4), int((3**jj) * n/4)] = lumin[ii, jj, 0]
            elif num_obj[ii, jj] == 2:
                # create two objects
                field_s[int((3**ii) * n / 4), int((3**jj) * n / 4)] = lumin[ii, jj, 0]
                field_s[int(((3**ii) * n) / 4), int(((3**jj) * n) / 4) + 2] = lumin[ii, jj, 1]

    # concatenate scene info into single array
    scene_info = np.concatenate((num_obj.reshape(4, 1), lumin.reshape(4, 2)), 1)

    return field_s, x, scene_info


# generate single object with specified size
def gen_sourcefield_obj(n, sz, obj_sz, obj_xoff, obj_yoff, lu):
    # create field coordinates
    x = np.linspace(-sz / 2, sz / 2, n + 1)

    # initialize source field array
    field_s = np.zeros((n + 1, n + 1))

    xms, yms = np.meshgrid(x, x)
    mask = np.sqrt((xms-obj_xoff)**2 + (yms-obj_yoff)**2) < obj_sz
    field_s[mask] = 1

    return field_s, x


# fresnel propagation utilizing fft
def fresnel_prop(field_s, xs_crd, zo, lam):
    # source coordinates
    ds = xs_crd[1] - xs_crd[0]
    ns = np.size(xs_crd)
    xms, yms = np.meshgrid(xs_crd, xs_crd)

    # receive coordinates
    xr_crd = np.linspace(-(1 / ds) / 2, (1 / ds) / 2, ns)
    xmr, ymr = np.meshgrid(xr_crd, xr_crd)

    # fresnel integral
    quadratic = np.exp(1j * np.pi * (xms ** 2 + yms ** 2) / (zo * lam))
    a = np.exp(1j * 2 * np.pi * zo / lam) * np.exp(1j * np.pi * (xmr ** 2 + ymr ** 2) / (lam * zo))
    field_r = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(quadratic * field_s))) / ns**2
    field_r = a * field_r / (1j * lam * zo)

    return field_r, xr_crd * (lam * zo)


# execute simplified version fraunhofer propagation
def fraunhofer_prop(field_s, xs_crd, zo, lam, focus_flag):
    # source coordinates
    ds = xs_crd[1] - xs_crd[0]
    ns = np.size(xs_crd, 0)
    xr_crd = np.linspace(-(1 / ds) / 2, (1 / ds) / 2, ns)
    xmr, ymr = np.meshgrid(xr_crd, xr_crd)

    # fourier transform and do a element-wise multiplication
    fs_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(field_s))) / ns**2
    if focus_flag:
        # amplitude term for the fraunhofer propagation
        a = np.exp(1j * np.pi * (xmr ** 2 + ymr ** 2) / (lam * zo))
        field_r = a * fs_fft / (1j * lam * zo)
    else:
        # amplitude term for the fraunhofer propagation
        a = np.exp(2 * 1j * np.pi * zo / lam) * np.exp(1j * np.pi * (xmr ** 2 + ymr ** 2) / (lam * zo))
        field_r = a * fs_fft / (1j * lam * zo)

    return field_r, xr_crd * (lam * zo)


# use rayleigh sommerfeld propagation on given source field
def rayleigh_sommerfeld_prop(field_s, xs_crd, zo, lam, n_r, sz_r, xc, yc):
    xs_mgrid, ys_mgrid = np.meshgrid(xs_crd, xs_crd)  # meshgrids for source x & y

    # Troubleshooting: use different method to generate coordinates
    # xr_crd = np.arange(-np.floor(sz_r / 2), np.ceil(sz_r / 2)) * dx_r + xc  # array of received x-coordinates
    # yr_crd = np.arange(-np.floor(sz_r / 2), np.ceil(sz_r / 2)) * dx_r + yc  # array of received y-coordinates

    xr_crd = np.arange(-sz_r / 2, (sz_r / 2) + (sz_r / n_r), sz_r / n_r) + xc
    yr_crd = np.arange(-sz_r / 2, (sz_r / 2) + (sz_r / n_r), sz_r / n_r) + yc

    field_r = np.zeros((np.size(xr_crd), np.size(yr_crd))) + 0j  # create receiving field (complex) with zeros

    # perform propagation
    phs_cent_idx = np.ceil(sz_r / 2)
    for ii in range(np.size(yr_crd)):
        for jj in range(np.size(xr_crd)):
            r = np.sqrt((xs_mgrid - xr_crd[jj]) ** 2 + (ys_mgrid - yr_crd[ii]) ** 2 + zo ** 2)
            field_r[ii, jj] = np.sum(
                np.sum(np.multiply(field_s, np.exp(1j * 2 * np.pi * r / lam) / (r ** 2 * lam * 1j))))
            if ii == phs_cent_idx and jj == phs_cent_idx:
                phs_center = 2 * np.pi * r / lam

    return field_r, xr_crd, phs_center


# Fraunhofer criteria
def fresnel_criteria(x, y, zeta, eta, lam):
    return np.cbrt(np.pi * ((x - zeta) ** 2 + (y - eta) ** 2) ** 2 / (4 * lam))


# Fresnel criteria
def fraunhofer_criteria(zeta, eta, lam):
    return np.pi * (zeta ** 2 + eta ** 2) / lam


# circle center circle masking, simulating a subreflector
def circ_mask(n, sz, outer_d, inner_d):
    # coordinates
    x = np.linspace(-sz / 2, sz / 2, n +1)
    xm, ym = np.meshgrid(x, x)
    mask = np.zeros((n + 1, n + 1))

    # outer diameter mask
    mask[(xm ** 2 + ym ** 2) <= (outer_d / 2) ** 2] = 1

    # inner diameter mask
    mask[(xm ** 2 + ym ** 2) < (inner_d / 2) ** 2] = 0

    return mask


# account for the points measured at focal plane array
def sq_mask(n, sz, x_diam, y_diam):
    # coordinates
    x = np.linspace(-sz / 2, sz / 2, n + 1)
    xm, ym = np.meshgrid(x, x)
    mask = np.zeros((n + 1, n + 1))

    # mask in x direction
    maskx = np.abs(xm) <= x_diam / 2

    # mask in y direction
    masky = np.abs(ym) <= y_diam / 2

    # combine mask
    maski = maskx * masky
    mask[maski] = 1

    return mask


# make an otf of a pupil function with inner radius (radius2_px) and outer radius (radius1_px)
def make_otf(radius1_px, radius2_px, source_px, scale, phase):
    aperture = np.zeros((source_px, source_px))
    coords = np.linspace(-source_px/2, source_px/2, source_px)
    xmat, ymat = np.meshgrid(coords, coords)
    distance_sq = (xmat**2 + ymat**2)

    aperture[distance_sq <= radius1_px**2] = 1
    aperture[distance_sq < radius2_px**2] = 0

    pupil = aperture * np.cos(phase) + 1j * (aperture * np.sin(phase))
    psf = np.real(np.fft.fft2(pupil)*np.conj(np.fft.fft2(pupil)))
    psf = scale * psf / np.sum(pupil.flatten())

    print('done')
    return np.fft.fft2(psf), aperture


# create phase screen from zernike polynomials
def zern_phase_scrn(ro, d, nn, zern_mx, x_crd, windx, windy, boil, deltat, frames, zern, ch, atm_flag):
    if atm_flag:
        # generate multiple phase screens over time using atmospheric vars
        screens = atmos_phase_screen(nn, ro, x_crd, windx, windy, boil, deltat, zern_mx, zern, ch, frames)
    else:
        # generate phase screen directly from zernike poly
        screens = general_phase_screen(nn, zern_mx, zern, ch)

    return screens


# generates phase screens randomly from zernike polynomials
def general_phase_screen(nn, zern_mx, zern, ch):
    # create random numbers to scale zernike polynomials
    rn = np.random.normal(size=(zern_mx - 1, 1))
    z_cf = np.matmul(ch, rn)

    # initialize zernike phase screen
    zern_phs = np.zeros((nn + 1, nn + 1))

    # summations of randomly scaled zernike polynomials
    for ii in np.arange(0, zern_mx - 1):
        zern_phs = zern_phs + z_cf[ii] * zern[ii, :, :]

    return zern_phs


# creates phase screens based on zernike polynom,ials and atmospheric variables
def atmos_phase_screen(nn, ro, x_crd, windx, windy, boil, deltat, zern_mx, zern, ch, frames):
    a = 6.88

    # get phase structure
    xm, ym = np.meshgrid(x_crd, x_crd)  # tau_x, tau_y
    phs_struct = a * (((ym + (windy + boil) * deltat) ** 2 + (xm + (windx + boil) * deltat) ** 2) ** (5 / 6) -
                      (xm ** 2 + ym ** 2) ** (5 / 6)) / ro ** (5 / 3)

    # denominator, Zernike sum of squares
    dnm = np.zeros((zern_mx - 1))
    for xx in np.arange(0, zern_mx - 1):
        dnm[xx] = np.sum(np.sum(zern[xx, :, :] ** 2))

    # FFT of all zernikes
    fft_mat = np.zeros((nn + 1, nn + 1, zern_mx - 1)) + 0j
    for jj in np.arange(0, zern_mx - 1):
        fft_mat[:, :, jj] = np.fft.fft2(np.fft.fftshift(zern[jj, :, :]))

    # inner double sum integral
    idsi = np.zeros((zern_mx - 1, zern_mx - 1))
    for xx in np.arange(0, zern_mx - 1):
        for yy in np.arange(0, zern_mx - 1):
            xcorr_fft = np.real(np.fft.fftshift(np.fft.ifft2(fft_mat[:, :, xx] * fft_mat[:, :, yy].conj())))
            idsi[xx, yy] = np.sum(np.sum(xcorr_fft * phs_struct / (dnm[xx] * dnm[yy])))
            # For Troubleshooting: check xcorr results
            # xcorr = signal.correlate2d(zern[xx, :, :], zern[yy, :, :])
            # idsi[xx, yy] = np.sum(np.sum(xcorr * phs_struct / (dnm[xx] * dnm[yy])))

    # get n structure function from the phase structure function differences
    phi = la.inv(ch)
    dn = np.zeros(zern_mx-1)
    temp = np.zeros((zern_mx-1, zern_mx-1, zern_mx-1))
    for ii in range(0, zern_mx-1):
        temp[:, :, ii] = np.outer(phi[ii, :], phi[ii, :])
        dn[ii] = np.sum(np.sum(idsi * temp[:, :, ii]))

    # get the n-vector, and correlation functions
    r_0 = 1
    r_n = r_0 - dn/2
    r_n = np.clip(r_n, a_min=0, a_max=None).reshape((1, zern_mx-1))
    n_vec = np.random.normal(size=(1, zern_mx - 1))
    cond_var = 1 - r_n**2
    cond_var = cond_var.reshape((1, zern_mx-1))

    # generate screens from statistics (update frame based on conditional mean and variance)
    z_record = np.zeros((zern_mx-1, frames))
    screens = np.zeros((nn+1, nn+1, frames))
    for ii in range(0, frames):
        atm_lens_phs = np.zeros((nn + 1, nn + 1))
        z_scale = ch @ n_vec.T
        z_record[:, ii] = np.squeeze(z_scale)
        for jj in np.arange(0, zern_mx - 1):
            atm_lens_phs = atm_lens_phs + z_scale[jj] * zern[jj, :, :]
        screens[:, :, ii] = atm_lens_phs
        cond_mean = n_vec * r_n
        n_vec = np.sqrt(cond_var) * np.random.normal(size=(zern_mx-1)) + cond_mean

    # check what the screens look like (when flags == True) for toubleshooting only
    check_screens_flag = False
    create_animation_flag = False

    # plot first four frames of phase screen
    if check_screens_flag:
        step = int(frames/4)  # assumes number of frames divisible by 4
        fig, ax = plt.subplots(4, step)
        for ii in range(0, 4):
            for jj in range(0,  step):
                ax[ii, jj].imshow(screens[:, :, jj + ii*4])
                ax[ii, jj].axes.xaxis.set_visible(False)
                ax[ii, jj].axes.yaxis.set_visible(False)
        plt.tight_layout()
        plt.show()

    # create and save animation as animated gif
    if create_animation_flag:
        fig = plt.figure()
        def AnimationFunction(f):
            plt.imshow(screens[:, :, f])
        ani = FuncAnimation(fig, AnimationFunction, frames=frames, interval=50)
        writer = PillowWriter(fps=10)
        ani.save('screens.gif', writer=writer)

    return screens


# takes in the zernike polynomials and creates covariance matrix and its cholesky decomp
def generate_zern_polys(zern_mx, nn, d, ro):
    k = 2.2698

    # create zernicke
    zern, idx = zern_poly(zern_mx + 1, nn)
    zern = zern[1: zern_mx, :, :]

    # transfer indices
    n = idx[:, 0]
    m = idx[:, 1]
    p = idx[:, 2]

    # calculate covariance matrix
    covar = np.zeros((zern_mx, zern_mx))
    for xx in np.arange(0, zern_mx):
        for yy in np.arange(0, zern_mx):
            test1 = m[xx] == m[yy]
            test2 = m[xx] == 0
            temp_frac = (p[xx] / 2) / np.ceil(p[xx] / 2)
            p_even = temp_frac == 1
            temp_frac = (p[yy] / 2) / np.ceil(p[yy] / 2)
            p_p_even = temp_frac == 1
            test3 = p_even == p_p_even
            test0 = test2 | test3
            if test1 and test0:
                k_zz = k * np.power(-1, (n[xx] + n[yy] - (2 * m[xx])) / 2) * np.sqrt((n[xx] + 1) * (n[xx] + 1))
                num = k_zz * math.gamma((n[xx] + n[yy] - (5 / 3)) / 2) * np.power(d / ro, 5 / 3)
                dnm = math.gamma((n[xx] - n[yy] + 17 / 3) / 2) * math.gamma((n[yy] - n[xx] + 17 / 3) / 2) * \
                      math.gamma((n[xx] + n[yy] + 23 / 3) / 2)
                covar[xx, yy] = num / dnm

    # factorize covariance matrix using cholesky
    covar = covar[1:zern_mx, 1:zern_mx]
    ch = la.cholesky(covar)

    return zern, ch


# create zernike polynomials from zernike indexes
def zern_poly(i_mx, num_pts):
    # define coordinates
    del_x = (1 / num_pts) * 2
    x_crd = del_x * np.linspace(int(-num_pts / 2), int(num_pts / 2), int(num_pts + 1))
    xm, ym = np.meshgrid(x_crd, x_crd)
    rm = np.sqrt(xm ** 2 + ym ** 2)
    thm = np.arctan2(ym, xm)

    # get zernike indexes from CSV file
    if i_mx > 1000:
        print('ERROR: TOO MANY ZERNIKE POLYNOMIALS REQUESTED')
    zern_idx = zern_indexes()
    zern_idx = zern_idx[0:i_mx - 1, :]

    # create array of 2d zernike polynomials with zernike radial function
    zern = np.zeros((int(i_mx), int(num_pts + 1), int(num_pts + 1)))
    for ii in np.arange(0, i_mx - 1):
        nn = zern_idx[ii, 0]
        mm = zern_idx[ii, 1]
        if mm == 0:
            zern[ii, :, :] = np.sqrt(nn + 1) * zrf(nn, 0, rm)
        else:
            if np.mod(ii, 2) == 0:
                zern[ii, :, :] = np.sqrt(2 * (nn + 1)) * zrf(nn, mm, rm) * np.cos(mm * thm)
            else:
                zern[ii, :, :] = np.sqrt(2 * (nn + 1)) * zrf(nn, mm, rm) * np.sin(mm * thm)
        mask = (xm ** 2 + ym ** 2) <= 1
        zern[ii] = zern[ii] * mask

    return zern, zern_idx


# pull zernike indices from csv file, defaults to zern_idx.csv
def zern_indexes(filename: str = 'zern_idx.csv'):
    root = Path(__file__).parent

    # read in csv file and convert to list
    raw = csv.DictReader(open(Path(root, filename)))
    raw_list = list(raw)

    # initialize z array
    r = 1000  # number of indixes in zern.csv file
    c = 3  # x, y, i
    z = np.zeros((r, c))

    # get 'x' and 'y' column values as index 'i'
    for row in np.arange(0, r):
        row_vals = raw_list[row]
        z[row, 0] = float(row_vals['x'])
        z[row, 1] = float(row_vals['y'])
        z[row, 2] = float(row_vals['i'])

    return z


# zernike radial function
def zrf(n, m, r):
    rr = 0
    for ii in np.arange(0, (n - m + 1) / 2):
        num = (-1) ** ii * math.factorial(n - ii)
        dnm = math.factorial(ii) * math.factorial(((n + m) / 2) - ii) * math.factorial(((n - m) / 2) - ii)
        rr = rr + (num / dnm) * r ** (n - (2 * ii))
    return rr
