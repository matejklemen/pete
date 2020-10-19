import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="explain_nlp",
    version="0.0.1",
    author="Matej Klemen",
    author_email="matej.klemen1337@gmail.com",
    description="Peek under the hood of NLP models with methods such as IME, LIME and their improvements",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matejklemen/explain_nlp",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)