from setuptools import setup

setup(
    name='popodds',
    version='0.0.1',
    description='popodds',
    long_description='popodds',
    long_description_content_type='text/x-rst',
    url='https://github.com/mdmould/popodds',
    author='Matthew Mould',
    author_email='mattdmould@gmail.com',
    license='MIT',
    packages=['popodds'],
    install_requires=['numpy', 'scipy', 'kalepy'],
    python_requires='>=3.7',
    )

