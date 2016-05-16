cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package( OpenCV REQUIRED )
add_executable( graph_separation graph_separation.cpp )
target_link_libraries( graph_separation ${OpenCV_LIBS} )

find_package( OpenCV REQUIRED )
add_executable( graph_separation1 graph_separation1.cpp )
target_link_libraries( graph_separation1 ${OpenCV_LIBS} )

