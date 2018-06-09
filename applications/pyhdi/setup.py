from setuptools import setup

if __name__ == '__main__':
    setup(
        name='pyhdi',
        version='0.0.3',
        packages=['pyhdi'],
        package_dir={'pyhdi': 'swig/python/'},
        package_data={'': ['_pyhdi.so']},
        include_package_data=True,
        license='MIT',
        author='Alle Veenstra',
        author_email='alle.veenstra@gmail.com'
    )
