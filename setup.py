from setuptools import setup, find_packages

setup(
    name='DynamicallyGrowingTransformer',
    version='0.1.0',
    author='Max Ploner',
    author_email='growing_transformer@maxploner.de',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},  # Optional
    package_data={"src/growing_transformer": ["py.typed"]},
    description='Dynamically Growing Transformer Using Firefly Neural Architecture Descent',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        "torch",
        "transformers",
        "pytest",
        "mypy",
        "matplotlib"
    ],
)
