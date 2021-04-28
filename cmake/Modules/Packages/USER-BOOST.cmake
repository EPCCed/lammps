find_package(Boost)
target_compile_definitions(lammps PRIVATE -DLAMMPS_BOOST)
target_link_libraries(lammps PRIVATE ${Boost_LIBRARIES})