from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
README = path.join(here, 'README.md')

# with open(path.join(here, 'README.rst'), encoding = 'utf-8') as f:
#   long_description = f.read()

try:
  from pypandoc import convert

  def read_md(f):
    return convert(f, 'rst')

except ImportError:
  convert = None
  print("warning: pypandoc not found, could not convert markdown to rst")
  def read_md(f):
    return open(f, 'r').read()


setup(
  name = 'pycaret',
  version = '0.0.1.dev1',

  description = 'pycaret 0.0.1:  a python for framework for classification and regression training',
  long_description = read_md(README),

  url = '',

  author = 'Philip Goddard',
  author_email = 'pmgoddard89@gmail.com',

  license = 'MIT',

  classifiers = [
  'Development Status :: 3 - Alpha',
  'Intended Audience :: Science/Research',
  'License :: OSI Approved :: MIT License',
  'Operating System :: OS Independent'
  'Programming Language :: Python :: 3.5'
  ],

  keywords = 'classification regression training',

  packages = find_packages(exclude = ['contrib', 'docs', 'tests*']),

  install_requires = ['numpy>=1.12.1',
                      'pandas>=0.19.2',
                      'scikit-learn>= 0.18.1',
                      'matplotlib>=2.0.0',
                      'seaborn>=0.7.1',
                      'scipy>=0.18.1']

)
