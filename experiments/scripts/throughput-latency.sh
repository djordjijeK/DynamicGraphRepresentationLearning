#!/bin/bash

# script options
clean_build=True

# execution options
walk_model="deepwalk"             # deepwalk | node2vec
paramP=0.5                        # node2vec paramP
paramQ=2.0                        # node2vec paramQ
sampler_init_strategy="weight"    # random | burnin | weight
declare -a graphs=("cora-graph")
declare -a walks_per_node=(10)
declare -a walk_length=(80)
range_search="true"               # range search mode
determinism="true"               # determinism

# create the data dir
mkdir -p data/latency_throughput/

# 1. convert graphs in adjacency graph format if necessary
for graph in "${graphs[@]}"; do
  FILE=../data/"${graph}".adj
  if test -f "$FILE"; then
      echo 'Skipping conversion of a graph ' "${graph[@]}" ' to adjacency graph format!'
  else
      echo 'Converting graph ' "${graph[@]}" ' to adjacency graph format ...'
      ./../bin/SNAPtoAdj -s -f "../data/${graph}" "../data/${graph}.adj"
      echo 'Graph ' "${graph[@]}" ' converted to adjacency graph format!'
  fi
done

# 2. build the executable
mkdir -p ../../build;
cd ../../build;
cmake -DCMAKE_BUILD_TYPE=Release ..;
cd experiments;
make throughput-latency

# 3. execute experiments
for wpv in "${walks_per_node[@]}"; do
    for wl in "${walk_length[@]}"; do
        for graph in "${graphs[@]}"; do
            printf "\n"
            printf "Graph: ${graph} \n"
            ./throughput-latency -s -f "data/${graph}.adj" -w "${wpv}" -l "${wl}" -model "${walk_model}" -paramP "${paramP}" -paramQ "${paramQ}" -init "${sampler_init_strategy}" -rs "${range_search}" -d "${determinism}" | tee data/latency_throughput/${graph}-${walk_model}.txt
        done
    done
done

# 4. clean build if necessary
if [ "$clean_build" = True ] ; then
    cd ../../;
    rm -rf build;
#    rm experiments/data/*.adj
fi