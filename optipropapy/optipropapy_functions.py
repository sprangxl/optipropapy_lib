# general python libraries
import math
import csv

# external libraries
import numpy as np
import numpy.linalg as la
from pathlib import Path


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

    # generate object with specific size and location
    xms, yms = np.meshgrid(x, x)
    mask = np.sqrt((xms-obj_xoff)**2 + (yms-obj_yoff)**2) < obj_sz
    field_s[mask] = lu

    return field_s, x


# create object with pixel at a specific point (offset from center)
def gen_sourcefield_pts(n, x_off, y_off, lu):
    # create field coordinates
    x_c = int(n / 2)
    y_c = int(n / 2)

    # initialize source field array
    field_s = np.zeros((n + 1, n + 1))

    # generate object with specific size and location
    field_s[int(y_c + y_off), int(x_c + x_off)] = lu

    return field_s


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
    # generate an aperture
    aperture = np.zeros((source_px, source_px))
    coords = np.linspace(-source_px/2, source_px/2, source_px)
    xmat, ymat = np.meshgrid(coords, coords)
    distance_sq = (xmat**2 + ymat**2)
    aperture[distance_sq <= radius1_px**2] = 1
    aperture[distance_sq < radius2_px**2] = 0

    # generate pupil and psf from aperture
    pupil = aperture * np.exp(1j * phase)
    pupil_ifft = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(pupil)))
    psf = np.real(pupil_ifft * np.conj(pupil_ifft))  # multiply by conjugate is real valued, 'np.real' for correct dtype
    psf = scale * psf / np.sum(psf.flatten())

    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psf))), aperture


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
    z_coef = np.matmul(ch, rn)

    # initialize zernike phase screen
    zern_phs = np.zeros((nn + 1, nn + 1))

    # summations of randomly scaled zernike polynomials
    for ii in np.arange(0, zern_mx - 1):
        zern_phs = zern_phs + z_coef[ii] * zern[ii, :, :]

    return zern_phs


# creates phase screens based on zernike polynom,ials and atmospheric variables
def atmos_phase_screen(nn, ro, x_crd, windx, windy, boil, deltat, zern_mx, zern, ch, frames):
    a = 6.88

    # increase size of the zernike polynomial spaces to allow for correlation calcs
    zern2 = np.zeros((zern_mx - 1, nn + 1, nn + 1))
    zern2[:, int(nn / 4):int(3 * nn / 4), int(nn / 4):int(3 * nn / 4)] = zern

    # get phase structure
    xm, ym = np.meshgrid(x_crd, x_crd)  # tau_x, tau_y
    phs_struct = a * (((ym + (windy + boil) * deltat) ** 2 + (xm + (windx + boil) * deltat) ** 2) ** (5 / 6) -
                      (xm ** 2 + ym ** 2) ** (5 / 6)) / ro ** (5 / 3)

    # denominator, Zernike sum of squares
    dnm = np.zeros((zern_mx - 1))
    for xx in np.arange(0, zern_mx - 1):
        dnm[xx] = np.sum(np.sum(zern2[xx, :, :] ** 2))

    # FFT of all zernikes
    fft_mat = np.zeros((nn + 1, nn + 1, zern_mx - 1)) + 0j
    for jj in np.arange(0, zern_mx - 1):
        fft_mat[:, :, jj] = np.fft.fft2(np.fft.fftshift(zern2[jj, :, :]))

    # inner double sum integral
    idsi = np.zeros((zern_mx - 1, zern_mx - 1))
    for xx in np.arange(0, zern_mx - 1):
        for yy in np.arange(0, zern_mx - 1):
            xcorr_fft = np.real(np.fft.fftshift(np.fft.ifft2(fft_mat[:, :, xx] * fft_mat[:, :, yy].conj())))
            idsi[xx, yy] = np.sum(np.sum(xcorr_fft * phs_struct / (dnm[xx] * dnm[yy])))

    # get n structure function from the phase structure function differences
    phi = la.inv(ch)
    d_n = np.zeros(zern_mx-1)
    phi_out = np.zeros((zern_mx-1, zern_mx-1, zern_mx-1))
    for ii in range(0, zern_mx-1):
        phi_out[:, :, ii] = np.outer(phi[ii, :], phi[ii, :])
        d_n[ii] = np.sum(np.sum(idsi * phi_out[:, :, ii]))

    # get the n-vector, and correlation functions
    r_0 = 1
    r_n = r_0 - d_n/2
    r_n = np.clip(r_n, a_min=0, a_max=1).reshape((1, zern_mx-1))
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
            atm_lens_phs = atm_lens_phs + z_scale[jj] * zern2[jj, :, :]
        screens[:, :, ii] = atm_lens_phs
        cond_mean = n_vec * r_n
        n_vec = np.sqrt(cond_var) * np.random.normal(size=(zern_mx-1)) + cond_mean

    return screens


# takes in the zernike polynomials and creates covariance matrix and its cholesky decomp
def generate_zern_polys(zern_mx, nn, d, ro):
    k = 2.2698

    # create zernike
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
                k_zz = k * (-1)**((n[xx] + n[yy] - (2 * m[xx])) / 2) * np.sqrt((n[xx] + 1) * (n[yy] + 1))
                num = k_zz * math.gamma((n[xx] + n[yy] - (5 / 3)) / 2) * (d / ro)**(5 / 3)
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


def distort_and_focus(source, otf, z, lam, x_coord):
    # get varaibles of field coordinates
    nn = np.size(x_coord)
    ll = np.max(x_coord) - np.min(x_coord)

    # get the light field as seen at the optic
    source_fft = np.fft.fft2(source)
    field = source_fft * otf

    # focus the field to the detector
    scale_fact = 1 / (lam * z)
    image = scale_fact * np.fft.ifft2(field)

    # get the new coordinates at the detector
    ds = ll / nn + 1
    xr_coord = np.linspace(-(1 / ds) / 2, (1 / ds) / 2, nn + 1) * (lam * z)

    return image, xr_coord


# takes in data, 2D intensity array, and its (number of iterations)
def gerschberg_saxton_phase_retrieve(data, apt, its, phs_init):
    point = np.sqrt(np.abs(data))
    pupil = apt
    pupil_phs = pupil * np.exp(1j * phs_init)
    point_phs = np.fft.fft2(pupil_phs)

    # iterate GS
    for it in range(its):
        pupil_cmplx = np.abs(pupil) * np.exp(1j * np.angle(pupil_phs))
        point_phs = np.fft.fft2(pupil_cmplx)
        point_cmplx = np.abs(point) * np.exp(1j * np.angle(point_phs))
        pupil_phs = np.fft.ifft2(point_cmplx)

    pupil_phs = np.angle(pupil_phs)
    psf_phs = np.angle(point_phs)
    return pupil_phs, psf_phs, np.abs(point_phs)


def convert_to_psf_otf(phs_screen):
    phase_fft = np.fft.fft2(phs_screen)
    psf = np.abs(phase_fft)**2
    psf = psf / np.sum(psf.flatten())
    otf = np.fft.fft2(psf)

    return psf, otf


def maxlikelihood_deconvolution(nn, frames, data, defocus, its, gs_its, name, psf_t, phases_t, obj_t):
    # initialize variables
    ap_est = circ_mask(nn, 1, 0.5, 0)
    ap_est = np.fft.fftshift(ap_est)
    phs_est = np.array(defocus)
    psf_est, otf_est = convert_to_psf_otf(ap_est * np.exp(1j * phs_est))  # initial psf and otf estimates
    obj_est = np.ones((nn + 1, nn + 1))

    psf_est = np.repeat(np.expand_dims(psf_est, 2), frames, axis=2)
    otf_est = np.repeat(np.expand_dims(otf_est, 2), frames, axis=2)
    phs_est = np.repeat(np.expand_dims(phs_est, 2), frames, axis=2)
    bias_est = np.median(data.flatten())

    img_est = np.zeros((nn + 1, nn + 1, frames))
    ratio_fft_est = np.zeros((nn + 1, nn + 1, frames)) + 0j
    obj_updt = np.zeros((nn + 1, nn + 1, frames))
    psf_updt = np.zeros((nn + 1, nn + 1, frames))

    # run iteration
    for ii in range(its):
        obj_fft_est = np.fft.fft2(obj_est)
        for ff in range(frames):
            # estimate the image function
            img_est[:, :, ff] = np.real(np.fft.ifft2(obj_fft_est * otf_est[:, :, ff])) + bias_est

            # the fourier transform of the data to image est ratio
            ratio_fft_est[:, :, ff] = np.fft.fft2(data[:, :, ff] / img_est[:, :, ff])

            # estimate the update for the object function
            obj_updt[:, :, ff] = np.real(np.fft.ifft2(ratio_fft_est[:, :, ff] * otf_est[:, :, ff].conj()))

            # estimate the update for the psf
            psf_updt[:, :, ff] = np.real(np.fft.ifft2(ratio_fft_est[:, :, ff] * obj_fft_est.conj()))

            # get the new psf from the update
            psf_est[:, :, ff] = np.abs(np.fft.ifft2(otf_est[:, :, ff])) * psf_updt[:, :, ff]
            psf_est[:, :, ff] = psf_est[:, :, ff] / np.sum(psf_est[:, :, ff].flatten())

            # estimate the phase of the phase of the psf
            phs_est[:, :, ff], pupil_phs, point_est = \
                gerschberg_saxton_phase_retrieve(psf_est[:, :, ff], ap_est, gs_its, phs_est[:, :, ff])

            # convert phase of the psf to the otf estimate
            psf_gs_est, otf_est[:, :, ff] = convert_to_psf_otf(ap_est * np.exp(1j * phs_est[:, :, ff]))

        print(name + f' Iteration: {ii+1}/{its}')

        obj_est = obj_est * np.sum(obj_updt, 2) / frames

    return obj_est, img_est, otf_est


def known_psf_deconvolution(nn, frames, data, its, otf_real, name):
    obj_est = np.ones((nn + 1, nn + 1))  # object estimate
    bias_est = np.median(data.flatten())

    img_est = np.zeros((nn + 1, nn + 1, frames))
    ratio_fft_est = np.zeros((nn + 1, nn + 1, frames)) + 0j
    obj_updt = np.zeros((nn + 1, nn + 1, frames))
    for ii in range(its):
        obj_fft_est = np.fft.fft2(obj_est)
        for ff in range(frames):
            # estimate the image function
            img_est[:, :, ff] = np.real(np.fft.ifft2(obj_fft_est * otf_real[:, :, ff])) + bias_est
            # the fourier transform of the data to image est ratio
            ratio_fft_est[:, :, ff] = np.fft.fft2(data[:, :, ff] / img_est[:, :, ff])
            # estimate the update for the object function
            obj_updt[:, :, ff] = np.real(np.fft.ifft2(ratio_fft_est[:, :, ff] * otf_real[:, :, ff].conj()))

        obj_est = obj_est * np.sum(obj_updt, 2) / frames
        print(name + f' Iteration: {ii+1}/{its}')

    return obj_est, img_est

