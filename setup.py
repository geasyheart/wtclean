import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wtclean",
    version="0.1.0",
    description="Wind turbine SCADA power-curve data cleaning via iterative bin + MAD filtering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(exclude=["test", "examples"]),
    python_requires=">=3.8",
    install_requires=["numpy", "pandas", "matplotlib"],
    keywords=["power-curve-filtering", "scada", "wind-energy", "outlier-detection"],
)
