SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_VERSION 1)


#Define cross compiler locations
SET(CMAKE_C_COMPILER ${TOOLCHAIN_DIR}/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/arm-linux-gnueabihf-gcc)
SET(CMAKE_CXX_COMPILER ${TOOLCHAIN_DIR}/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/bin/arm-linux-gnueabihf-c++)

#Define sysroot path for Raspberry-Pi distribution for cross compiling 
SET(CMAKE_FIND_ROOT_PATH ${TOOLCHAIN_DIR}/arm-bcm2708/arm-rpi-4.9.3-linux-gnueabihf/arm-linux-gnueabihf/sysroot)

#Use our definitions for compiler tools 
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

add_definitions(-std=c++14 -mfpu=neon)
