from setuptools import find_packages, setup

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="DynamicallyGrowingTransformer",
    version="0.1.0",
    author="Max Ploner",
    author_email="growing_transformer@maxploner.de",
    packages=find_packages(where="src"),
    package_dir={"": "src"},  # Optional
    include_package_data=True,
    description="Dynamically Growing Transformer Using Firefly Neural Architecture Descent",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=install_requires,
)
