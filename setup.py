from setuptools import find_packages, setup

setup(
    name='optipropapy',
    packages=find_packages(include=['optipropapy']),
    version='0.1.0',
    description='statistical fourier optics library',
    author='Joshua Sprang',
    license='MIT',
    setup_requires=['numpy', 'scipy', 'pathlib', 'matplotlib'],
    test_requires=['pytest==4.4.1'],
    test_suite='tests'
)