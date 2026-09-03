import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wtclean",
    version="0.1.0",
    description="Wind turbine SCADA power-curve data cleaning via iterative bin + MAD filtering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="BSD-3-Clause",
    license_files=["LICENSE"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    packages=setuptools.find_packages(exclude=["test", "examples"]),
    python_requires=">=3.10",
    install_requires=["numpy", "pandas", "matplotlib"],
    keywords=[
        "wind-energy",
        "wind-turbine",
        "scada",
        "power-curve",
        "outlier-detection",
        "data-cleaning",
    ],
)
