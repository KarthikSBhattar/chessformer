from setuptools import find_packages, setup

setup(
    name='chessformer',
    packages=find_packages(include=['chessformer']),
    version='0.1.0',
    url='https://github.com/KarthikSBhattar/chessformer',
    description='The Chessformer Engine',
    author='Karthik',
    install_requires=[
        'absl-py',
        'apache-beam',
        'chess',
        'chex',
        'dm-haiku',
        'jax',
        'jaxtyping',
        'jupyter',
        'numpy',
        'optax',
        'orbax-checkpoint',
        'pandas',
        'scipy',
        'typing-extensions',
        'uvicorn',
    ],
)