import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='genepro',
    version='1.7.0',
    author='Marco Virgolin',
    author_email='marco.virgolin@cwi.nl',
    url='https://github.com/giorgia-nadizar/genepro',
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    setup_requires=['wheel'],
    install_requires=[
    	'numpy>=2.1.1',
    	'sympy>=1.13.2',
        'scikit-learn>=1.5.1',
        'joblib>=1.4.2',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11.9',
)
