from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
d = generate_distutils_setup(
    packages=['arbitration'],
    package_dir={'': 'src/Shared_control_architecture'}
)
setup(**d)
