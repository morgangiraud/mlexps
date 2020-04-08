import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name="ml_experiments",
    version="0.1.0",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    license="Apache License 2.0",
    url="https://github.com/morgangiraud/ml_experiments",
    description="Random ML experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Morgan Giraud",
    author_email="contact@morganigraud.com"
)
