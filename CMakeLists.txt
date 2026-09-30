cmake_minimum_required(VERSION 3.16)
project(vision_training LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

add_executable(test_environment src/common/test_environment.cpp)
target_include_directories(test_environment PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(test_environment PRIVATE ${OpenCV_LIBS} Eigen3::Eigen Ceres::ceres)

add_executable(task1 src/task1_image/main.cpp)
target_include_directories(task1 PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(task1 PRIVATE ${OpenCV_LIBS} Eigen3::Eigen Ceres::ceres)

add_executable(task2 src/task2_fit/main.cpp)
target_include_directories(task2 PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(task2 PRIVATE ${OpenCV_LIBS} Eigen3::Eigen Ceres::ceres)

add_executable(task3 src/task3_windmill/main.cpp)
target_include_directories(task3 PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(task3 PRIVATE ${OpenCV_LIBS} Eigen3::Eigen Ceres::ceres)
