# -------------------------- DYNAMIC GRAPH REPRESENTATION LEARNING WITH METROPOLIS HASTINGS ----------------------------

option(CLANG     OFF)
option(CILK      OFF)
option(OPENMP    OFF)
option(HOMEGROWN OFF)
option(SERIAL    OFF)

# compile flags options
set(OMPFLAGS     "-DOPENMP -fopenmp")
set(CILKFLAGS    "-DCILK -fcilkplus")
set(HGFLAGS      "-DHOMEGROWN -pthread")
set(OPTIMIZATION "-O3 -g")
set(CFLAGS       "-DEDGELONG  -mcx16 ${OPTIMIZATION} -ldl -std=c++17 -march=native -Wall -Wno-subobject-linkage -DUSEMALLOC")

if(CLANG)
    set(CMAKE_CXX_COMPILER "clang++")
    set(PFLAGS ${CILKFLAGS})
elseif(CILK)
    set(CMAKE_CXX_COMPILER "g++")
    set(PFLAGS ${CILKFLAGS})
elseif(OPENMP)
    set(CMAKE_CXX_COMPILER "g++")
    set(PFLAGS ${OMPFLAGS})
elseif(HOMEGROWN)
    set(CMAKE_CXX_COMPILER "g++")
    set(PFLAGS ${HGFLAGS})
elseif(SERIAL)
    set(CMAKE_CXX_COMPILER "g++")
    set(PFLAGS "")
else()
    set(CMAKE_CXX_COMPILER "g++")
    set(PFLAGS ${HGFLAGS})
endif()

# set compile flags
set(CMAKE_CXX_FLAGS "${CFLAGS} ${PFLAGS}")

file(COPY data DESTINATION .)
file(COPY bin/SNAPtoAdj DESTINATION .)

# experiments executables
add_executable(throughput-latency src/throughput-latency.cpp)  ## src/throughput-latency.cpp
add_executable(memory-throughput-latency src/memory-throughput-latency.cpp)

# -------------------------- DYNAMIC GRAPH REPRESENTATION LEARNING WITH METROPOLIS HASTINGS ----------------------------