#ifndef DYNAMIC_GRAPH_REPRESENTATION_LEARNING_WITH_METROPOLIS_HASTINGS_COMPRESSED_WALKS_H
#define DYNAMIC_GRAPH_REPRESENTATION_LEARNING_WITH_METROPOLIS_HASTINGS_COMPRESSED_WALKS_H

#include <graph/tree_plus/edge_plus.h>
#include <graph/tree_plus/walk_plus.h>

namespace dynamic_graph_representation_learning_with_metropolis_hastings
{
    /**
    * CompressedWalks is a structure that stores graph random walks in a compressed format.
    * It essentially represents a compressed purely functional tree (C-Tree) that achieves compression
    * based on differential coding.
    */
    class CompressedWalks : public walk_plus::treeplus
    {
        public:
            /**
             * Caching the current {min, max} for vnext in a walk-tree for the search in range optimization
             */
            types::Vertex vnext_min;
            types::Vertex vnext_max;

            /**
             * @brief CompressedWalks default constructor.
             */
            CompressedWalks() : walk_plus::treeplus(), vnext_min(0), vnext_max(0) {};

            /**
             * @brief CompressedWalks constructor.
             *
             * @param plus - tree plus
             * @param root - tree root
             */
            CompressedWalks(walk_plus::AT* plus, walk_plus::treeplus::Node* root, types::Vertex _vnext_min, types::Vertex _vnext_max)
                : walk_plus::treeplus(plus, root), vnext_min(_vnext_min), vnext_max(_vnext_max) {};

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
            CompressedWalks(const Sequence &sequence, types::Vertex source, types::Vertex _vnext_min, types::Vertex _vnext_max, pbbs::flags flag = pbbs::no_flag)
                : walk_plus::treeplus(sequence, source, flag), vnext_min(_vnext_min), vnext_max(_vnext_max) {};

            /**
            * @brief Finds the next vertex in the walk given walk id and position
            *
            * @param walk_id  - unique walk id
            * @param position - position in the walk (starts from 0)
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

                    auto this_walk_id  = pair.first / config::walk_length;                  // todo: needs floor?
                    auto this_position = pair.first - (this_walk_id * config::walk_length); // todo: position here starts from 0. verify this one!
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
            * @brief Finds the next vertex in the walk given walk id and position in a dedicated range
            *
            * @param walk_id  - unique walk id
            * @param position - position in the walk
            * @param source   - current walk vertex
            * @param low      - lower bound of the
            * @return - next vertex in the walk
            */
            types::Vertex find_next_in_range(types::WalkID walk_id, types::Position position, types::Vertex source)
            {
                types::Vertex next_vertex = -1;

                // Formula for encoding walk_id and position
                auto formula = walk_id * (config::walk_length) + position;
                // Bounds of the search range
                auto lb = pairings::Szudzik<size_t>::pair({formula, this->vnext_min});
                auto ub = pairings::Szudzik<size_t>::pair({formula, this->vnext_max});
                cout << "wid=" << walk_id << ", pos=" << (int) position << ", source=" << source << " and range [lb=" << lb << ", ub=" << ub << "]"
                     << ", with formula=" << formula << " vnext_min=" << this->vnext_min << ", and vnext_max=" << this->vnext_max << endl;

//                bool result = this->iter_elms_cond(source, [&](auto value)                    // O(n)
                bool result = this->iter_elms_cond_in_range(source, lb, ub, [&](auto value)  // O(blogn + k) output sensitive search
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
    };
}

#endif // DYNAMIC_GRAPH_REPRESENTATION_LEARNING_WITH_METROPOLIS_HASTINGS_COMPRESSED_WALKS_H
