cmake_minimum_required(VERSION 2.8.11)
project(naive_bayes)


# Build the particle filter project and solution.
# Use C++11
set(SRCS main.cpp classifier.cpp )
set_source_files_properties(${SRCS} PROPERTIES COMPILE_FLAGS -std=c++0x)

# Create the executable
add_executable(naive_bayes ${SRCS})


#set(SRCS src/main.cpp src/particle_filter_sol.cpp)
#set_source_files_properties(${SRCS} PROPERTIES COMPILE_FLAGS -std=c++0x)
#add_executable(naive_bayes ${SRCS})









