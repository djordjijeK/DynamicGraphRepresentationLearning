#ifndef DYNAMIC_GRAPH_REPRESENTATION_LEARNING_WITH_METROPOLIS_HASTINGS_COMPRESSED_WALKS_H
#define DYNAMIC_GRAPH_REPRESENTATION_LEARNING_WITH_METROPOLIS_HASTINGS_COMPRESSED_WALKS_H

#include "edge_plus.h"
#include "walk_plus.h" // todo: remove this. just included it to make sure the walk tree is included somewhere

namespace dynamic_graph_representation_learning_with_metropolis_hastings
{
    /**
    * CompressedWalks is a structure that stores graph random walks in a compressed format.
    * It essentially represents a compressed purely functional tree (C-Tree) that achieves compression
    * based on differential coding.
    */
    class CompressedWalks : public tree_plus::treeplus
    {
        public:
            struct edge_entry {
                using key_t = uintV; // the 'head' edge of this node.
                using val_t = AT*; // the set of edges stored in this node.
                static bool comp(const key_t& a, const key_t& b) { return a < b; }
                using aug_t = uintV; // num. edges in this subtree
                static aug_t get_empty() { return 0; }
                static aug_t from_entry(const key_t& k, const val_t& v) { return 1 + lists::node_size(v); }
                static aug_t combine(const aug_t& a, const aug_t& b) { return a + b; }
                using entry_t = std::pair<key_t,val_t>;
                static entry_t copy_entry(const entry_t& e) {
                    // TODO: Instead of copying, bump a ref-ct (note that copy_node and
                    // deallocate can implement these semantics internally)
                    return make_pair(e.first, lists::copy_node(e.second));
                }
                static void del(entry_t& e) {
                    if (e.second) {
                        // TODO: Should decrement ref-ct, free if only owner
                        lists::deallocate(e.second);
                    }
                }
            };


            /**
             * @brief CompressedWalks default constructor.
             */
            CompressedWalks() : tree_plus::treeplus() {};

            /**
             * @brief CompressedWalks constructor.
             *
             * @param plus - tree plus
             * @param root - tree root
             */
            CompressedWalks(tree_plus::AT* plus, tree_plus::treeplus::Node* root)
                : tree_plus::treeplus(plus, root) {};

            /**
            * @brief CompressedWalks constructor.
            *
            * @tparam Sequence
            *
            * @param sequence - sequence of walk parts
            * @param source   - graph vertex
            * @param flag
            */
            template<class Sequence>
            CompressedWalks(const Sequence &sequence, types::Vertex source, pbbs::flags flag = pbbs::no_flag)
                : tree_plus::treeplus(sequence, source, flag) {};

            /**
            * @brief Finds the next vertex in the walk given walk id and position
            * REMARK: This find operation in linear to the size of a walk-tree
            *
            * @param walk_id  - unique walk id
            * @param position - position in the walk
            * @param source   - current walk vertex
            *
            * @return - next vertex in the walk
            */
            types::Vertex find_next(types::WalkID walk_id, types::Position position, types::Vertex source)
            {
                types::Vertex next_vertex = -1;

                bool result = this->iter_elms_cond(source, [&](auto value)
                {
                    auto pair = pairings::Szudzik<types::Vertex>::unpair(value);

                    auto this_walk_id  = pair.first / config::walk_length;
                    auto this_position = pair.first - (this_walk_id * config::walk_length);
                    next_vertex        = pair.second;

                    if (this_walk_id == walk_id && this_position == position)
                        return true;
                    else
                        return false;
                });

                #ifdef MALIN_DEBUG
                    if (!result || next_vertex == -1)
                    {
                        std::cerr << "Dock debug error! CompressedWalks::FindNext::walk_id = "
                                  << walk_id
                                  << ", position = "
                                  << (int) position
                                  << ", vertex = "
                                  << source
                                  << std::endl;

                        std::exit(1);
                    }
                #endif

                return next_vertex;
            }


            /**
            * @brief Finds the next vertex in the walk given walk id and position in an optimized way
            * TODO: Implement the find_next_inrange using (1) pairing functions ordering properties and (2) traversal I did
            * REMARK: The optimized find next operation is a range tree search which has complexity O(blogn + k), where k items in range
            *
            * @param walk_id  - unique walk id
            * @param position - position in the walk
            * @param source   - current walk vertex
            *
            * @return - next vertex in the walk
            */
            types::Vertex find_next_inrange(types::WalkID walk_id, types::Position position, types::Vertex source)
            {
                TODO:

                return next_vertex;
            }
    };
}

#endif // DYNAMIC_GRAPH_REPRESENTATION_LEARNING_WITH_METROPOLIS_HASTINGS_COMPRESSED_WALKS_H
