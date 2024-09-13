from setuptools import setup, find_packages

setup(
    name="adtool",
    version="0.1.1",
    description="Libraries needed to run the autodisc-server computational container.",
    author="Jesse Lin, Zacharie Bugaud",
    author_email="jesse.lin@inria.fr, zacharie.bugaud@inria.fr",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9,<3.13",
    install_requires=[
        "addict==2.4.0",
        "filetype==1.2.0",
        "graphviz==0.20.1",
        "imageio==2.34.0",
        "imageio-ffmpeg==0.4.9",
        "matplotlib>=3.8",
        "mergedeep==1.3.4",
        "neat-python==0.92",
        "numpy>=1.26",
        "pexpect>=4.8.0",
        "pillow==10.2.0",
        "requests>=2",
        "sqlalchemy>=2.0",
        "tinydb==4.8.0",
        "toml>=0.10.2",
     #   "torch>=1.7.1",
        "urllib3==2.2.1",
        "watchdog>=4.0.0",
        "annotated-types==0.6.0",
        "pydantic==2.7.1",
        "moviepy==1.0.3",
        "ipython==8.24.0",
    ],
    extras_require={
        "examples": [
            "transformers",
            "diffusers",
            "accelerate",
            "torch>=1.7.1",
            "scipy"
        ],
        "visu": [
            "websockets",
            "watchfiles",
            "scikit-learn",
            "opencv-python",
            "fastapi[uvicorn]",
            "umap-learn"
        ],
        "docking":[
            "rdkit",
            "selenium",
            "webdriver_manager",
            "imageio",
            "crem"
        ]
    },
    packages=find_packages(include=[".","adtool","adtool.*"]),
    package_dir={"": "."},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            # Define any command-line scripts here, if necessary.
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Change if you use a different license
        "Operating System :: OS Independent",
    ],
    tests_require=[
        # pytest or other testing libraries if needed
    ],
)
