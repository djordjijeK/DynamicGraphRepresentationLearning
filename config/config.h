#ifndef DYNAMIC_GRAPH_REPRESENTATION_LEARNING_WITH_METROPOLIS_HASTINGS_CONFIG_H
#define DYNAMIC_GRAPH_REPRESENTATION_LEARNING_WITH_METROPOLIS_HASTINGS_CONFIG_H

#define dygrl dynamic_graph_representation_learning_with_metropolis_hastings

#include <utility.h>
#include <types.h>
#include <globals.h>

auto graph_update_time_on_insert = timer("GraphUpdateTimeOnInsert", false);
auto walk_update_time_on_insert  = timer("WalkUpdateTimeOnInsert", false);
// --- profiling
//auto walk_insert_init            = timer("WalkInsertTimeForInitialization", false);
auto Walking_new_sampling_time           = timer("WalkInsertTimeFor2Jobs", false);
auto Walking_insert_new_samples           = timer("WalkInsertTimeFor2Accs", false);
//auto ij                          = timer("WalkInsertJobTime", false);
//auto dj                          = timer("WalkDeleteJobTime", false);
//auto walk_find_in_vertex_tree    = timer("FindInVertexTreeTime", false);
//auto walk_find_next_tree         = timer("FindNextTime", false);
auto szudzik_hash_insert                = timer("SzudzikTimeInsert", false);
auto szudzik_hash_delete                = timer("SzudzikTimeDelete", false);

auto MAV_time                    = timer("MAVTime", false);
// --- profile find next in range
//auto fnir_tree_search            = timer("FNIR_TreeSearchTime", false);
//auto read_access_MAV             = timer("ReadAccessMAV", false);
// ---
//auto bdown_create_vertex_entries = timer("VertexEntries", false);
//auto apply_multiinsert_ctrees    = timer("ApplyAccumulators", false);
//auto linear_cuckoo_acc_scann     = timer("LinearCuckooAccScan", false);

auto graph_update_time_on_delete = timer("GraphUpdateTimeOnDelete", false);
auto walk_update_time_on_delete  = timer("WalkUpdateTimeOnDelete", false);

auto merge_calc_triplets_to_delete = timer("MergeCalcTripletsToDelete", false);
auto merge_create_delete_walks     = timer("CreateDeleteWalks", false);
auto merge_multiinsert_ctress      = timer("MergeMultiinsertCtrees", false);
// ---
//auto ij_sampling                   = timer("IJsampling", false);
//auto ij_szudzik                    = timer("IJszudzik", false);
// ---
//auto mav_deletions_obsolete        = timer("MAVdeletions", false);
//auto mav_iteration                 = timer("MAViteration", false);
// ---
auto Merge_time                      = timer("MergeAllTimer", false);
//auto sortAtMergeAll                = timer("SortAtMergeAll", false);
//auto accumultinsert                = timer("Accumultinsert", false);
auto LastMerge                     = timer("LastMerge", false);
// ---
auto ReadWalks                     = timer("ReadWalks", false);
auto FindPreviousVertexNode2Vec    = timer("FindPreviousVertexNode2Vec", false); // Time to rewalk from the first node till the end
auto FindNextNodeSimpleSearch      = timer("FindNextSimpleSearch", false);
auto FindNextNodeRangeSearch       = timer("FindNextRangeSearch", false);

// Min and Max Measurements
auto MAV_min = 1000.0;
auto MAV_max = 0.0;
auto Merge_min = 1000.0;
auto Merge_max = 0.0;
auto WalkSampling_min = 1000.0;
auto WalkSampling_max = 0.0;
auto WalkInsert_min = 1000.0;
auto WalkInsert_max = 0.0;

#endif // DYNAMIC_GRAPH_REPRESENTATION_LEARNING_WITH_METROPOLIS_HASTINGS_CONFIG_H
