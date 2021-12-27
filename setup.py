from setuptools import find_packages, setup
setup(
    name='molecular-dynamics-engine',
    packages=find_packages(),
    version='1.0',
    description='A simple molecular dynamics engine',
    author='Morris',
    license='MIT',
    install_requires=['numpy', 'scipy', 'matplotlib', 'jupyter',
                      'scipy', 'tqdm']
)