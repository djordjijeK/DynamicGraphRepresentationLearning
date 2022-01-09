#ifndef DYNAMIC_GRAPH_REPRESENTATION_LEARNING_WITH_METROPOLIS_HASTINGS_CONFIG_H
#define DYNAMIC_GRAPH_REPRESENTATION_LEARNING_WITH_METROPOLIS_HASTINGS_CONFIG_H

#define dygrl dynamic_graph_representation_learning_with_metropolis_hastings

#include <utility.h>
#include <types.h>
#include <globals.h>

auto graph_update_time_on_insert = timer("GraphUpdateTimeOnInsert", false);
auto walk_update_time_on_insert  = timer("WalkUpdateTimeOnInsert", false);
// --- profiling
auto walk_insert_init            = timer("WalkInsertTimeForInitialization", false);
auto walk_insert_2jobs           = timer("WalkInsertTimeFor2Jobs", false);
auto walk_insert_2accs           = timer("WalkInsertTimeFor2Accs", false);
auto ij                          = timer("WalkInsertJobTime", false);
auto dj                          = timer("WalkDeleteJobTime", false);
auto walk_find_in_vertex_tree    = timer("FindInVertexTreeTime", false);
auto walk_find_next_tree         = timer("FindNextTime", false);
auto szudzik_hash                = timer("SzudzikTime", false);
// --- profile find next in range
auto fnir_tree_search            = timer("FNIR_TreeSearchTime", false);
// ---
auto MAV_time                    = timer("MAV_time", false);

auto graph_update_time_on_delete = timer("GraphUpdateTimeOnDelete", false);
auto walk_update_time_on_delete  = timer("WalkUpdateTimeOnDelete", false);

#endif // DYNAMIC_GRAPH_REPRESENTATION_LEARNING_WITH_METROPOLIS_HASTINGS_CONFIG_H
