from setuptools import setup, find_packages
import os

VERSION = '1.0'
DESCRIPTION = 'Graph many mathematical functions'
working_directory = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(working_directory, "README.md"), encoding='utf-8') as f:
    long_description1 = f.read()
# Setting up
setup(
    name="marksfuncs",
    version=VERSION,
    author="mark. (Marc Pérez)",
    author_email="<marcperezcarrasco2010@gmail.com>",
    url='https://github.com/marc1fino/marksfuncs',
    description=DESCRIPTION,
    long_description=long_description1,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib', 'scipy'],
    license='MIT',
    keywords=['python', 'maths', 'graphics', 'functions', 'xy', 'mathematical functions'],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)