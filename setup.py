import pathlib
from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / 'README.md').read_text()

setup(
    name='paranet',
    version='0.1.4',    
    description='paranet package',
    long_description=README,
    long_description_content_type="text/markdown",
    url='https://github.com/ErikinBC/paranet',
    author='Erik Drysdale',
    author_email='erikinwest@gmail.com',
    license='GPLv3',
    license_files = ('LICENSE.txt'),
    packages=['paranet'],
    package_data={'paranet': ['/*','multivariate/*','univariate/*','tests/*', 'examples/*']},
    include_package_data=True,
    install_requires=['numpy', 'pandas', 'scipy'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
    ],
)
