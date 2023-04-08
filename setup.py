from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
  name="stalingrad",
  version="0.1.0",
  author="Maciej Kilian",
  author_email="kilianmaciej6@gmail.com",
  description="our autograd engine",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/iejMac/stalingrad",
  install_requires=["numpy", "requests"],
  packages=find_packages(),
  python_requires='>=3.8',
)
