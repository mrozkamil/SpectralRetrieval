from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()
        
setup(name='doppler',
      version='0.1.9.1',
      description='tools to analyse and simulate the radar Doppler spectra',
      url='http://github.com/...',
      author='Kamil Mroz',
      author_email='kamil.mroz@le.ac.uk',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      include_package_data=True,
      install_requires=[
          'numpy',
          'scipy',
	  'matplotlib',
	  'netCDF4'])
