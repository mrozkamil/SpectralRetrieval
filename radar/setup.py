from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()
        
setup(name='radar',
      version='0.2.6.6',
      description='tools to analyse and simulate the radar data',
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
	  'matplotlib'])
