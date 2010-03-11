#Define the project we are working on
project(discretemscomplex)

# Look for various software we need
cmake_minimum_required(VERSION 2.6)

set(QT_MIN_VERSION "4.5.0")
find_package(Qt4 REQUIRED)

if(NOT QT4_FOUND)
  message(FATAL_ERROR "Failed to find Qt 4.5 or greater.")
endif(NOT QT4_FOUND)

set(QT_USE_QTGUI TRUE)
set(QT_USE_QTOPENGL TRUE)

include(${QT_USE_FILE})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Boost COMPONENTS regex thread)

if(NOT Boost_FOUND)
  message(FATAL_ERROR "Cannot find boost")
endif(NOT Boost_FOUND)

find_package(OpenCL)

if(NOT OPENCL_FOUND)
  message("could not find openCL ..In linux \n
	  set LD_LIBRARY_PATH to point to the sdk's lib path .. \n
	  Ill find the rest")
endif(NOT OPENCL_FOUND)
 

add_subdirectory(glapp_framework)

set(dmsc_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/src/main.cpp 

	      ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/include/discreteMorseDS.h         
	      ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/include/discreteMorseAlgorithm.h  
	      ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/include/discreteMorseDSRenderFuncs.h  	      
	      
	      ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/include/rectangle_complex.h  
	      
	      ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/include/quad_datamanager.h           	      
	      ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/src/quad_datamanager.cpp  
	      
	      ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/include/quad_datamanager_ui.h  	      
	      ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/src/quad_datamanager_ui.cpp  
	      
	      ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/include/quad_dataset.h
	      ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/src/quad_dataset.cpp
	      
	      ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/include/grid_dataset.h
	      ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/src/grid_dataset.cpp
	      
	      ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/include/grid_mscomplex.h
	      ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/src/grid_mscomplex.cpp	      
	      
	      ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/include/grid_datamanager.h  
	      ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/src/grid_datamanager.cpp
	      
	      ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/include/grid_datamanager.h  
	      ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/src/grid_datamanager_ui.cpp  	      
	      )

file(GLOB ply_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/ply/*.c ${CMAKE_CURRENT_SOURCE_DIR}/ply/*.h)

QT4_WRAP_CPP(dmsc_MOC_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/include/quad_datamanager_ui.h			   
			   ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/include/quad_datamanager.h
			   ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/include/grid_datamanager_ui.h
			   ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/include/grid_datamanager.h)

file(GLOB dmsc_UIS ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/forms/quad_datamanager_frame.ui
		   ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/forms/grid_datamanager_frame.ui)

QT4_ADD_RESOURCES(dmsc_RC_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/resources/mscomplex.qrc)

QT4_WRAP_UI(dmsc_UI_SRCS ${dmsc_UIS})


include_directories(
  ${OPENCL_INCLUDE_DIRS}
  ${QT_ADDITIONAL_INCLUDE_PATHS}
  ${Boost_INCLUDE_DIRS}
  ${glapp_framework_SOURCE_DIR}/utls/include
  ${glapp_framework_SOURCE_DIR}/qtcolorpicker/include
  ${glapp_framework_SOURCE_DIR}/framework/include
  ${CMAKE_CURRENT_SOURCE_DIR}/dmsc/include
  ${CMAKE_CURRENT_SOURCE_DIR}/ply
  ${CMAKE_CURRENT_BINARY_DIR}  
  )

add_executable(${PROJECT_NAME} ${dmsc_SRCS} ${dmsc_MOC_SRCS} ${dmsc_UI_SRCS} ${dmsc_RC_SRCS} ${ply_SRCS})

target_link_libraries(${PROJECT_NAME} glapp_framework ${Boost_LIBRARIES} ${OPENCL_LIBRARIES})