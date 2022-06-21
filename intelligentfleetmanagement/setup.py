import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fleetmanager-droidsagency",
    version="0.0.1",
    author="Droids Agency",
    description="Fleet management",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    package_data={"fleetmanager": ["dummy_data.sql", "dashboard/assets/style.css",],},
    python_requires=">=3.8",
    install_requires=[
        "requests~=2.2",
        "numpy~=1.2",
        "pandas~=1.3",
        "SQLAlchemy==1.3.24",
        "python-dotenv~=0.1",
        "dash-bootstrap-templates~=1.0",
        "numpy-financial~=1.0",
        "bokeh~=2.4",  # TODO: Get rid of this
    ],
)
