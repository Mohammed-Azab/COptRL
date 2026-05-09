from setuptools import setup, find_packages
from glob import glob

package_name = 'quarter_car_gazebo'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.py')),
        (f'share/{package_name}/urdf',   glob('urdf/*')),
        (f'share/{package_name}/worlds', glob('worlds/*')),
        (f'share/{package_name}/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'sim_node           = quarter_car_gazebo.sim_node:main',
            'gazebo_bridge_node = quarter_car_gazebo.gazebo_bridge_node:main',
        ],
    },
)
