from setuptools import setup

setup(
    name='mlem',
    version='0.0.1',
    packages=find_packages("mlem")
    install_requires=['numpy', 'Pillow'],
    tests_require=['pytest']
)
