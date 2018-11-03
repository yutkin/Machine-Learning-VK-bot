from setuptools import setup, find_packages

setup(
    name='vkbot-front',
    version="0.0.1",
    python_requires='>=3.7',
    packages=find_packages(exclude=['tests']),
    entry_points={
        'console_scripts': ['vkbot = vkbot.app:start_app'],
    },
    namespace_packages=['vkbot'],
)