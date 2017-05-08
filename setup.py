from setuptools import Command, Extension, setup


def define_extensions():

    compile_args = ['-ffast-math', '-march=native', '-std=c11']

    return [Extension("netrex.libpredict",
                      ['netrex/predict.c'],
                      extra_compile_args=compile_args)]


class BuildExtension(Command):

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):

        from netrex.native import _build_module

        _build_module()


setup(
    name='netrex',
    version='0.1.0',
    requirements=['pytorch==0.1.11'],
    packages=['netrex'],
    license='MIT',
    cmdclass={'build_extension': BuildExtension},
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence'],
    ext_modules=define_extensions()
)
