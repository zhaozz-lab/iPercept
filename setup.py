"""Setup module for project."""

from setuptools import setup, find_packages

setup(
        name='mp18-project-skeleton',
        version='0.1',
        description='Skeleton code for Machine Perception Eye Tracking project.',

        author='Seonwook Park',
        author_email='spark@inf.ethz.ch',

        packages=find_packages(exclude=[]),
        python_requires='>=3.5',
        install_requires=[
            'coloredlogs',
            'h5py',
            'numpy',
            'opencv-python',
            'pandas',
            'ujson',

            # Install the most appropriate version of Tensorflow
            # Ref. https://www.tensorflow.org/install/
            'tensorflow',
        ],
)
