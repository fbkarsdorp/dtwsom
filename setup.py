import setuptools
from distutils.core import setup 

setup(
    name='dtwsom',
    version='0.0.1',
    packages=['dtwsom'],
    url='http://fbkarsdorp.github.io/dtwsom',
    author='Folgert Karsdorp',
    author_email='fbkarsdorp AT fastmail DOT nl',
    install_requires=['numpy', 
                      'matplotlib',
                      'dtw',
                      'seaborn'],
    dependency_links=[
        "git+https://github.com/fbkarsdorp/dtw.git"])