echo "Building ROS nodes"

cd example/ROS/ECT_SLAM
rm -rf build
mkdir build
cd build
cmake .. -DROS_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3
make -j8
