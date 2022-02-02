# -------------------------- DYNAMIC GRAPH REPRESENTATION LEARNING WITH METROPOLIS HASTINGS ----------------------------
cmake_minimum_required(VERSION 3.13)
project(dynamic_graph_representation_learning_with_metropolis_hastings)

# libraries
include_directories(libs/libcuckoo)
include_directories(libs/compressed_trees)
include_directories(libs/cds)
#include_directories(libs/concurrentqueue)

# include paths
include_directories(graph)
include_directories(walks)
include_directories(utils)
include_directories(config)

#find_library(tbb)
#find_package(TBB COMPONENTS concurrent_unordered_set)
#find_package(TBB)
#if (TBB_FOUND)
#    include_directories(${TBB_INCLUDE_DIRS})
#    set(LINK_LIBS ${LINK_LIBS} ${TBB_LIBRARIES})
#    set(HAVE_INTEL_TBB TRUE)
#endif()

# set up google tests
enable_testing()
add_subdirectory(tests)

# experiments
add_subdirectory(experiments)
# -------------------------- DYNAMIC GRAPH REPRESENTATION LEARNING WITH METROPOLIS HASTINGS ----------------------------