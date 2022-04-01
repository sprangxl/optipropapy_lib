from optipropapy import optipropapy_functions as opp
import matplotlib.pyplot as plt
import numpy as np


def test_optipropapy():
    # scene variables
    lam = 5e-9  # 5nm wavelength
    size = 2  # length of single spatial dimension in source field
    samples = 160  # number of samples in a single spatial dimension (functions assume even # of samples given)
    lumens = 1  # max brightness of randomly generated light sources
    min_percent = 0.5  # min percent of the max brightness an object can be generated as
    distance = 1e6  # distance to source field
    c = 2.998e8  # speed of light

    # atmosphere variables
    zern_max = 100  # max number of zernikes used
    r0 = 0.05  # fried's seeing parameter
    windx = 1  # wind in x direction
    windy = 1  # wind in y direction
    boil = 1  # atmospheric boil factor
    time_delta = 0.01  # exposure time per image frame
    frames = 1  # number of image frames
    atm_flag = True  # if false, generic phase screen defaults to 1 frame instead of array of frames

    # optic variables
    opt_diameter = 0.3  # diameter of the optic
    sub_diameter = 0.05  # diameter of a sub-optic (can be zero)
    focal_length = 2  # focal length of the optic

    # generate a source field, source field coordinates, and scene info/meta-data
    field_source, x_coordinates, scene_info = opp.gen_sourcefield_random(samples, size, lumens, min_percent)

    # propagate source field using a propagation method (example: fresnel propagation)
    field_receive, coordinates_rx = opp.fresnel_prop(field_source, x_coordinates, distance, lam)

    # generate the zernike polynomials prior to generating atmosphere phase screens
    zern, chol = opp.generate_zern_polys(zern_max, samples, opt_diameter, r0)

    # generate phase screens using the zernike polynomials
    screens = opp.zern_phase_scrn(r0, opt_diameter, samples, zern_max, coordinates_rx,
                                  windx, windy, boil, time_delta, frames,
                                  zern, chol, atm_flag)

    # generate a telescope mask function
    size_receive = 2 * max(coordinates_rx)  # get size of the receive field for mask function
    mask = opp.circ_mask(samples, size_receive, opt_diameter, sub_diameter)

    # element multiple field at optic by the mask and phase screen (note, screens = [samples x samples x frames])
    field_distorted = field_receive * mask * np.exp(-1j * (lam * c) * screens[:, :, 0])

    # focus receive field using fraunhofer propagation
    field_result, coordinates_rslt = opp.fraunhofer_prop(field_distorted, coordinates_rx, focal_length, lam, False)

    # display resultant field magnitude
    coord_min_rx = min(coordinates_rx)  # get coordinate's min to specify in extent
    coord_max_rx = max(coordinates_rx)  # get coordinate's max to specify in extent
    field_mag_rx = np.abs(field_receive)  # magnitude of received field
    field_mag_dstd = np.abs(field_distorted)  # magnitude of distorted field

    coord_min_rslt = min(coordinates_rslt)  # get coordinate's min to specify in extent
    coord_max_rslt = max(coordinates_rslt)  # get coordinate's max to specify in extent
    field_mag_rslt = np.abs(field_result)  # magnitude of resultant field

    # define and display plots
    fig, ax = plt.subplots(2, 3)

    # original source field
    ax[0, 0].imshow(field_source, extent=[-size/2, size/2, size/2, -size/2])
    ax[0, 0].set(title='Original (Source) Field')
    # received field
    ax[0, 1].imshow(field_mag_dstd, extent=[coord_min_rx, coord_max_rx, coord_max_rx, coord_min_rx])
    ax[0, 1].set(title='Received (@Optic) Field')
    # resultant field
    ax[0, 2].imshow(field_mag_rslt, extent=[coord_min_rslt, coord_max_rslt, coord_max_rslt, coord_min_rslt])
    ax[0, 2].set(title='Final (Resultant) Field')
    # atmosphere
    ax[1, 0].imshow(np.real(screens[:, :, 0]), extent=[coord_min_rx, coord_max_rx, coord_max_rx, coord_min_rx])
    ax[1, 0].set(title='Atmosphere Phase Screen')
    # mask
    ax[1, 1].imshow(mask, extent=[coord_min_rx, coord_max_rx, coord_max_rx, coord_min_rx])
    ax[1, 1].set(title='Optic Mask Function')
    # distorted field
    ax[1, 2].imshow(field_mag_rx, extent=[coord_min_rx, coord_max_rx, coord_max_rx, coord_min_rx])
    ax[1, 2].set(title='Original Rx Field')
    plt.show()


if __name__ == '__main__':
    test_optipropapy()