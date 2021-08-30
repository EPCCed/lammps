#CGAL_Qt5 is needed for the drawing and CGAL_Core is needed for this special Kernel.
find_package(CGAL REQUIRED COMPONENTS Qt5 Core)
if(CGAL_FOUND AND CGAL_Qt5_FOUND)
    #required to use basic_viewer
    add_definitions(-DCGAL_USE_BASIC_VIEWER -DQT_NO_KEYWORDS)
    #link it with the required CGAL libraries
    target_link_libraries(lammps PRIVATE CGAL::CGAL CGAL::CGAL_Qt5 CGAL::CGAL_Core)
else()
    message("ERROR: this program requires CGAL and CGAL_Qt5 and will not be compiled.")
endif()


SET(PB_LIB /home/james/cpp_packages/BallPivoting/build)
message(AUTHOR_WARNING ${PB_LIB})
find_library(BALLPIVOT NAMES ballpivotlib PATHS ${PB_LIB} NO_DEFAULT_PATH)
message(AUTHOR_WARNING ${BALLPIVOT})
if(BALLPIVOT)
    target_compile_definitions(lammps PRIVATE -DBALLPIVOT)
    target_link_libraries(lammps PRIVATE "${BALLPIVOT}")
    target_include_directories(lammps PRIVATE /home/james/cpp_packages/BallPivoting/src)
    message("Ball pivot found")
else()
    message("ERROR: this program requires BALL_PIVOT.")
endif()