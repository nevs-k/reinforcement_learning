from setuptools import setup, find_packages

setup(
    name='highway-env1',
    version='1.0.dev0',
    description='parking-v1',
    url='https://github.com/nevs-k/reinforcement_learning',
    author_email='nevsk@web.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='autonomous highway driving simulation environment reinforcement learning',
    packages=find_packages(exclude=['docs', 'scripts', 'tests']),
    install_requires=['gym', 'numpy', 'pygame', 'matplotlib', 'pandas'],
    tests_require=['pytest'],
    extras_require={
        'dev': ['scipy'],
        'deploy': ['pytest-runner', 'sphinx<1.7.3', 'sphinx_rtd_theme']
    },
    entry_points={
        'console_scripts': [],
    },
)