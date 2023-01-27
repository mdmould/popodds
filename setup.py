from setuptools import setup, find_packages

name = 'popodds'
version = '0.2.2'

with open('README.md', 'r') as f:
    long_description = f.read().strip()

setup(
    name=name,
    version=version,
    description='Simple package for Bayesian model comparison.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=f'https://github.com/mdmould/{name}',
    author='Matthew Mould',
    author_email='mattdmould@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy', 'scipy'],
    python_requires='>=3.7',
    )

