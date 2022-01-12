from setuptools import find_packages, setup

setup(
    name='AttentionVisualizer',
    packages=find_packages(include=['AttentionVisualizer']),
    version='0.1.0',
    description='Visualize important words in a text using the self-attention scores.',
    author='AlaFalaki',
    license='MIT',
    install_requires=['torch', 'ipywidgets', 'nltk', 'transformers']
)