OPTI-PROPA-PY includes a list of functions for using fourier optics to propagate light fields.

Functions (optipropapy/optipropapy_functions.py) include:

GENERATE RANDOM DATA

    gen_source_field

PROPAGATION FUNCTIONS:
    
    fresnel_prop

    fraunhofer_prop

    rayleigh_sommerfeld_prop

    fresnel_criteria

    fraunhofer_criteria

MASKING FUNCTIONS:

    circ_mask

    sq_mask

ZERNIKE PHASE SCREEN FUNCTIONS (Tabs indicate sub-functions):

    zern_phase_scrn
        general_phase_screen
        atmos_phase_screen

    generate_zern_polys
        zern_poly
            zern_indexes
            zrf

Test script (tests/optipropapy_tests.py) runs an example scenario using the functions in the library# optipropapy_lib
