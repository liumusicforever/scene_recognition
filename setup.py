from setuptools import find_packages, setup


setup(
    name='scr',
    version='0.1.0',
    description='Scene Recognition Library',
    include_package_data=True,
    packages=["scr", "scr.matchers"]
)