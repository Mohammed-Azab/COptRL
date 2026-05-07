from setuptools import setup, find_packages

setup(
    name='quarter_car_core',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.11.0',
        'gymnasium>=0.29.0',
        'matplotlib>=3.7.0',
        'pyyaml>=6.0',
        'pillow>=10.0.0',
    ],
)
