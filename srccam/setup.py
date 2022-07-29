import os
import setuptools


def parse_requirements():
    here = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(here, 'requirements.txt')) as f:
        lines = f.readlines()
    lines = [line for line in map(lambda line: line.strip(), lines) if line!='' and line[0] !='#']
    return lines

setuptools.setup(
    name='srccam',
    version="1.2.1",
    author="author",
    author_email="xxx@mail.com",
    description="Package for projection and read calibs",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        # "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: MIT License",
        # "Operating System :: OS Independent",
    ],
    install_requires=parse_requirements(),
)
