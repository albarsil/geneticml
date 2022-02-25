from setuptools import find_packages, setup


def readme():
    with open('README.md') as f:
        return f.read()
 
setup(
    name='geneticml',
    version='1.0.5',
    description='A simple and lightweight genetic algorithm for optimization of any machine learning model',
    long_description=readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/albarsil/geneticml',
    author='Allan Barcelos',
    author_email='albarsil@gmail.com',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    keywords=['machine learning', 'genetic', 'sklearn', 'pytorch', 'data science'],
    packages=find_packages(exclude=['tests.*', 'tests']),
    install_requires=['tqdm'],
    test_suite='tests.test_suite'
)
