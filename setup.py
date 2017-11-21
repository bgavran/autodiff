from setuptools import setup

setup(name='automatic_differentiation',
      version='0.1',
      description='Rudimentary automatic differentiation framework',
      url='https://github.com/bgavran/autodiff',
      author='Bruno Gavranović',
      author_email='bgavran3@gmail.com',
      packages=['automatic_differentiation', 'automatic_differentiation.core', 'automatic_differentiation.visualization'],
      license='MIT',
      install_requires=['numpy', 'graphviz'],
      )
