from setuptools import setup, find_packages

setup(
    name='mlem',
    version='0.0.1',
    packages=["mlem"],
    install_requires=['numpy', 'Pillow', 'scipy'],
    tests_require=['pytest'],
    license='MIT'
)
