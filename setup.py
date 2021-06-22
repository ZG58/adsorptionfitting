import pathlib
from setuptools import find_packages, setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="curve_fit-gui",
    version="1.0.0",
    description="Gui for scipy curve_fit() function",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/", # need to add rep ref
    author="jskanger",
    author_email="j.s.kanger@utwente.nl", # something else?
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=["matplotlib", ], # need to add
    # add entrypoints?
)