import setuptools


setuptools.setup(
    name="mpc-rendezvous",
    version="0.0.1",
    author="Kyle Krol",
    author_email="kpk63@cornell.edu",
    description="A small set of model predictive controllers for spacecraft rendezvous",
    url="https://github.com/kylekrol/mpc-rendezvous",
    package_dir={"": "mpc"},
    packages=setuptools.find_packages(where="mpc"),
    python_requires=">=3.6",
)
