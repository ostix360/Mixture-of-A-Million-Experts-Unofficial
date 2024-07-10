from setuptools import setup, find_packages

requirements = [
    'numpy',
    'torch',
    'einops',
]

setup(
    name='peer',
    version='0.1',
    description='Unofficial implementation of the PEER model',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/ostix360/Mixture-of-A-Million-Experts-Unofficial",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.8",
    author='Ostix',
    packages=find_packages(),
    install_requires=requirements,
)
