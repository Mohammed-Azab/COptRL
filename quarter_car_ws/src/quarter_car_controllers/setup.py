from setuptools import setup, find_packages

package_name = 'quarter_car_controllers'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            [f'resource/{package_name}']),
        (f'share/{package_name}', ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'lqr_node = quarter_car_controllers.lqr_node:main',
            'mpc_node = quarter_car_controllers.mpc_node:main',
            'rl_node  = quarter_car_controllers.rl_node:main',
        ],
    },
)
