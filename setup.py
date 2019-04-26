from setuptools import setup

setup(name='autodiff',
      version='0.1',
      description='Rudimentary automatic differentiation framework',
      url='https://github.com/bgavran/autodiff',
      author='Bruno Gavranović',
      author_email='bruno@brunogavranovic.com',
      packages=['autodiff', 'autodiff.core', 'autodiff.visualization'],
      license='MIT',
      install_requires=['numpy', 'graphviz'],
      )
