#!/bin/bash

# script options
clean_build=True                                    # cleans build folder after the execution

# execution options
walk_model="deepwalk"                               # deepwalk | node2vec
paramP=0.5                                          # node2vec's paramP
paramQ=2.0                                          # node2vec's paramQ
sampler_init_strategy="random"                      # random | burnin | weight
declare -a graphs=("youtube-graph" "livejournal-graph" "orkut-graph")             # array of graphs
declare -a walks_per_vertex=(10)                    # walks per vertex to generate
declare -a walk_length=(80)                         # length of one walk
#declare -a sizes=(1 2 3 4 5 6 7 8 9 10 11 12)       # exponent of the head frequency
declare -a sizes=(8)       # exponent of the head frequency
range_search="true"                                 # range search mode
determinism="true"                                  # determinism

# create the data dir
#mkdir -p data/chunk_sizes/

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
make chunk-size

# 3. execute experiments
for wpv in "${walks_per_vertex[@]}"; do
    for wl in "${walk_length[@]}"; do
        for graph in "${graphs[@]}"; do
            for hf in "${sizes[@]}"; do
                printf "\n"
                printf "Graph: ${graph}\n"
                ./chunk-size -s -f "data/${graph}.adj" -w "${wpv}" -l "${wl}" -model "${walk_model}" -paramP "${paramP}" -paramQ "${paramQ}" -init "${sampler_init_strategy}" -rs "${range_search}" -d "${determinism}" -hf "${hf}" #| tee data/chunk_sizes/${graph}-${sizes}.dat
            done
        done
    done
done

# 4. clean build if necessary
if [ "$clean_build" = True ] ; then
    cd ../../;
    rm -rf build;
    # rm experiments/data/*.adj
fi
