from setuptools import setup, find_packages


setup(
    name="lost_souls_chat",
    version="0.1.0",
    author="Biraj Dahal",
    author_email="dahalbiraj@icloud.com",
    description="A tool for analyzing conversational data",
    long_description="A tool for analyzing conversational data",
    long_description_content_type="text/markdown",
    url="https://github.com/biraj-dahal/Loved-Lost-Souls",
    packages=find_packages(), 
    python_requires=">=3.9,<3.12",  
    install_requires=[
        "convokit>=3.1.0",
        "pandas>=2.2.3",
        "numpy>=1.26.4",
        "scikit-learn>=1.5.2",
        "transformers>=4.49.0",
        "torch>=2.4.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "lost-souls-chat=lost_souls_chat.main:lost_souls_chat",
        ],
    },
)