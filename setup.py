from setuptools import setup, find_packages

setup(
    name="GermanyWindCast",
    version="0.1",
    description="Code for Bachelor's Thesis of Quirin Brandl",
    author="Quirin Brandl",
    author_email="quirin.brandl17@gmail.com",
    url="",
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'windrose',
        'geopandas',
        'requests',
        'bs4',
        'scikit-learn'
    ],
    python_requires='>=3.6',
)