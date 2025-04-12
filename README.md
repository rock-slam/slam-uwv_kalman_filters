uwv_kalman_filters
=============

VelocityUKF
---
The focus of the VelocityUKF is to provide a model-aided linear and angular velocity estimate.  Due to the model-aiding the velocities can be provided in a high frequency.

PoseUKF
---
The PoseUKF is a model aided inertial localization solution for autonomous underwater vehicles. As minimal input the filter relays on rotation rates and accelerations from an IMU and velocities from a DVL.  Given force and torque measurements an AUV motion model aids the velocity estimate during DVL drop outs.  ADCP measurements further aid the estimation in cases of DVL bottom-lock loss. Given gyroscopes capable of sensing the rotation of the earth (e.g. a fiber optic gyro) the filter is able to estimate it's true heading.

Install and compile
-------

This C++ library uses cmake and can be compiled within ROCK, ROS or independently.
The following instructions have been tested on Ubuntu 16.04 (Xenial).

### Using ROCK
Make sure that the Ruby interpreter (recommended version 2.3) is installed on your machine.
#### Setup ROCK core bootstrap
```bash
mkdir rock
cd rock
wget http://www.rock-robotics.org/autoproj_bootstrap
ruby autoproj_bootstrap git https://github.com/rock-core/buildconf.git
source env.sh
autoproj update # All configurations can be answered with the default parameters besides of the Rock flavor. This must be set to master.
```
NOTE: Please select `master` as Rock flavor. All other parameters can be the defaults.

#### Checkout, build and install the library
```bash
amake slam/uwv_kalman_filters
```


### Using ROS kinetic
#### Install ROS
```bash
wget http://packages.ros.org/ros.key -O - | sudo apt-key add -
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-get update -qq
sudo apt-get install -y python-rosdep python-rosinstall python-vcstools
sudo rosdep init
rosdep update
```
#### Checkout repositories
Create directories
```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src/
```
Checkout repositories
```bash
wget https://raw.githubusercontent.com/rock-slam/slam-uwv_kalman_filters/ros_package_xml/.rosinstall
wstool update
```

#### Build and install
```bash
cd ..
source /opt/ros/kinetic/setup.bash
catkin init
catkin config --install --cmake-args -DBINDINGS_RUBY=OFF -DBUILD_SHARED_LIBS=ON
catkin build
```

NOTE: A ROS node is currently not available.


License
-------
BSD 3-Clause License


Rock Standard Layout
--------------------

This directory structure follows some simple rules, to allow for generic build
processes and simplify reuse of this project. Following these rules ensures that
the Rock CMake macros automatically handle the project's build process and
install setup properly.

```
STRUCTURE
-- src/ 
	Contains all header (*.h/*.hpp) and source files
-- build/
	The target directory for the build process, temporary content
-- bindings/
	Language bindings for this package, e.g. put into subfolders such as
   |-- ruby/ 
        Ruby language bindings
-- viz/
        Source files for a vizkit plugin / widget related to this library 
-- resources/
	General resources such as images that are needed by the program
-- configuration/
	Configuration files for running the program
-- external/
	When including software that needs a non standard installation process, or one that can be
	easily embedded include the external software directly here
-- doc/
	should contain the existing doxygen file: doxygen.conf
```
