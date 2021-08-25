#ifndef DYNAMIC_GRAPH_REPRESENTATION_LEARNING_WITH_METROPOLIS_HASTINGS_MALIN_H
#define DYNAMIC_GRAPH_REPRESENTATION_LEARNING_WITH_METROPOLIS_HASTINGS_MALIN_H

#include <graph/api.h>
#include <cuckoohash_map.hh>
#include <concurrentqueue.h>

#include <config.h>
#include <pairings.h>
#include <vertex.h>
#include <snapshot.h>

#include <models/deepwalk.h>
#include <models/node2vec.h>

namespace dynamic_graph_representation_learning_with_metropolis_hastings
{
    /**
     * @brief Malin represents a structure that stores a graph as an augmented parallel balanced binary tree.
     * Keys in this tree are graph vertices and values are compressed edges, compressed walks, and metropolis hastings samplers.
     */
    class Malin
    {
        public:
            using Graph = aug_map<dygrl::Vertex>;

            /**
             * @brief Malin constructor.
             *
             * @param graph_vertices - total vertices in a graph
             * @param graph_edges    - total edges in a graph
             * @param offsets        - vertex offsets for its neighbors
             * @param edges          - edges
             * @param free_memory    - free memory excess after graph is loaded
             */
            Malin(long graph_vertices, long graph_edges, uintE* offsets, uintV* edges, bool free_memory = true)
            {
                #ifdef MALIN_TIMER
                    timer timer("Malin::Constructor");
                #endif

                // 1. Initialize memory pools
                Malin::init_memory_pools(graph_vertices, graph_edges);

                // 2. Create an empty vertex sequence
                using VertexStruct = std::pair<types::Vertex, VertexEntry>;
                auto vertices = pbbs::sequence<VertexStruct>(graph_vertices);

                // 3. In parallel construct graph vertices
                parallel_for(0, graph_vertices, [&](long index)
                {
                    size_t off = offsets[index];
                    size_t deg = ((index == (graph_vertices - 1)) ? graph_edges : offsets[index + 1]) - off;
                    auto S = pbbs::delayed_seq<uintV>(deg, [&](size_t j) { return edges[off + j]; });

                    if (deg > 0)
                        vertices[index] = std::make_pair(index, VertexEntry(types::CompressedEdges(S, index),
                                                                            dygrl::CompressedWalks(),
                                                                            new dygrl::SamplerManager(0)));
                    else
                        vertices[index] = std::make_pair(index, VertexEntry(types::CompressedEdges(),
                                                                            dygrl::CompressedWalks(),
                                                                            new dygrl::SamplerManager(0)));
                });

                // 4. Construct the graph
                auto replace = [](const VertexEntry& x, const VertexEntry& y) { return y; };
                this->graph_tree = Graph::Tree::multi_insert_sorted(nullptr, vertices.begin(), vertices.size(), replace, true);

                // 5. Memory cleanup
                if (free_memory)
                {
                    pbbs::free_array(offsets);
                    pbbs::free_array(edges);
                }

                vertices.clear();

                #ifdef MALIN_TIMER
                    timer.reportTotal("time(seconds)");
                #endif
            }

            /**
             * @brief Number of vertices in a graph.
             *
             * @return - the number of vertices in a graph
             */
            [[nodiscard]] auto number_of_vertices() const
            {
                size_t n = this->graph_tree.size();
                auto last_vertex = this->graph_tree.select(n - 1);

                return n > 0 ? last_vertex.value.first + 1 : 0;
            }

            /**
             * @brief Number of edges in a graph.
             *
             * @return - the number of edges in a graph
             */
            [[nodiscard]] auto number_of_edges() const
            {
                return this->graph_tree.aug_val();
            }

            /**
             * @brief Flattens the vertex tree to an array of vertex entries.
             *
             * @return - the sequence of pointers to graph vertex entries
             */
            [[nodiscard]] FlatVertexTree flatten_vertex_tree() const
            {
                #ifdef MALIN_TIMER
                    timer timer("Malin::FlattenVertexTree");
                #endif

                types::Vertex n_vertices = this->number_of_vertices();
                auto flat_vertex_tree    = FlatVertexTree(n_vertices);

                auto map_func = [&] (const Graph::E& entry, size_t ind)
                {
                    const types::Vertex& key = entry.first;
                    const auto& value = entry.second;
                    flat_vertex_tree[key] = value;
                };

                this->map_vertices(map_func);

                #ifdef MALIN_TIMER
                    timer.reportTotal("time(seconds)");

                    std::cout << "Flat vertex tree memory footprint: "
                              << utility::MB(flat_vertex_tree.size_in_bytes())
                              << " MB = " << utility::GB(flat_vertex_tree.size_in_bytes())
                              << " GB" << std::endl;
                #endif

                return flat_vertex_tree;
            }

            /**
            * @brief Flattens the graph to an array of vertices, their degrees, neighbors, and sampler managers.
            *
            * @return - the sequence of vertices, their degrees, neighbors, and sampler managers
            */
            [[nodiscard]] FlatGraph flatten_graph() const
            {
                #ifdef MALIN_TIMER
                    timer timer("Malin::FlattenGraph");
                #endif

                size_t n_vertices = this->number_of_vertices();
                auto flat_graph   = FlatGraph(n_vertices);

                auto map_func = [&] (const Graph::E& entry, size_t ind)
                {
                    const uintV& key  = entry.first;
                    const auto& value = entry.second;

                    flat_graph[key].neighbors = entry.second.compressed_edges.get_edges(key);
                    flat_graph[key].degree    = entry.second.compressed_edges.degree();
                    flat_graph[key].samplers  = entry.second.sampler_manager;
                };

                this->map_vertices(map_func);

                #ifdef MALIN_TIMER
                    timer.reportTotal("time(seconds)");

                    auto size = flat_graph.size_in_bytes();

                    std::cout << "Malin::FlattenGraph: Flat graph memory footprint: "
                              << utility::MB(size)
                              << " MB = " << utility::GB(size)
                              << " GB" << std::endl;
                #endif

                return flat_graph;
            }

            /**
             * @brief Traverses vertices and applies mapping function.
             *
             * @tparam F
             *
             * @param map_f   - map function
             * @param run_seq - determines whether to run part of the code sequantially
             * @param granularity
             */
            template<class Function>
            void map_vertices(Function map_function, bool run_seq = false, size_t granularity = utils::node_limit) const
            {
                this->graph_tree.map_elms(map_function, run_seq, granularity);
            }

            /**
             * @brief Destroys malin instance.
             */
            void destroy()
            {
                this->graph_tree.~Graph();
                this->graph_tree.root = nullptr;
            }

            /**
             * @brief Destroys inverted index.
             */
            void destroy_index()
            {
                auto map_func = [&] (Graph::E& entry, size_t ind)
                {
                    // delete compressed walks - plus part
                    if (entry.second.compressed_walks.plus)
                    {
                        lists::deallocate(entry.second.compressed_walks.plus);
                        entry.second.compressed_walks.plus = nullptr;
                    }
                    // delete compresed walks - tree part
                    if (entry.second.compressed_walks.root)
                    {
                        auto T = walk_plus::edge_list();

                        T.root = entry.second.compressed_walks.root;
                        entry.second.compressed_walks.root = nullptr;
                    }

                    // delete the sample manager
                    entry.second.sampler_manager->clear();
                };

                this->map_vertices(map_func);
            }

            /**
            * @brief Creates initial set of random walks.
            */
            void generate_initial_random_walks()
            {
                auto graph             = this->flatten_graph();
                auto total_vertices    = this->number_of_vertices();
                auto walks_to_generate = total_vertices * config::walks_per_vertex;
                auto cuckoo            = libcuckoo::cuckoohash_map<types::Vertex, std::vector<types::Vertex>>(total_vertices); // todo: maybe PairedTriplet

                using VertexStruct  = std::pair<types::Vertex, VertexEntry>;
                auto vertices       = pbbs::sequence<VertexStruct>(total_vertices);

                // ------- Code for (min, max) of the next of each vertex -------------------------
                constexpr const size_t array_next_size = 20;
                types::Vertex next_min_stack[array_next_size], next_max_stack[array_next_size];
                types::Vertex* next_min = next_min_stack; types::Vertex* next_max = next_max_stack;
                if (total_vertices > array_next_size)
                {
                    next_min = pbbs::new_array<types::Vertex>(total_vertices);
                    next_max = pbbs::new_array<types::Vertex>(total_vertices);
                }
                // Initialize all mins to max integer 32 bits and all maxs to zero
                parallel_for(0, total_vertices, [&] (types::Vertex i) {
                    next_min[i] = std::numeric_limits<uint32_t>::max();
                    next_max[i] = 0;
                });
                // ---------------------------------------------------------------------------------

                RandomWalkModel* model;
                switch (config::random_walk_model)
                {
                    case types::DEEPWALK:
                        model = new DeepWalk(&graph);
                        break;
                    case types::NODE2VEC:
                        model = new Node2Vec(&graph, config::paramP, config::paramQ);
                        break;
                    default:
                        std::cerr << "Unrecognized random walking model" << std::endl;
                        std::exit(1);
                }

                // 1. walk in parallel
                parallel_for(0, walks_to_generate, [&](types::WalkID walk_id)
                {
                    // todo: this is for certain corner cases of node2vec. can stripe it out
                    if (graph[walk_id % total_vertices].degree == 0)
                    {
//                        types::PairedTriplet hash = pairings::Szudzik<types::Vertex>::pair({walk_id*config::walk_length,std::numeric_limits<uint32_t>::max() - 1});
                        types::PairedTriplet hash = pairings::Szudzik<types::Vertex>::pair({walk_id * config::walk_length + 0, walk_id % total_vertices});
                        cuckoo.insert(walk_id % total_vertices, std::vector<types::Vertex>());
                        cuckoo.update_fn(walk_id % total_vertices, [&](auto& vector) {
                            vector.push_back(hash);
                        });

                        // ---------------------------------------------------------------
                        // Refine the next (min, max) for the current vertex -------------
                        next_min[walk_id % total_vertices] = std::min(next_min[walk_id % total_vertices], walk_id % total_vertices);
                        next_max[walk_id % total_vertices] = std::max(next_max[walk_id % total_vertices], walk_id % total_vertices);
//                        cout << "--- first node with degree zero, next min=" << next_min[walk_id % total_vertices] << " and max=" << next_max[walk_id % total_vertices] << endl;
                        // ---------------------------------------------------------------

                        return;
                    }

                    auto random = utility::Random(walk_id / total_vertices);
                    types::State state  = model->initial_state(walk_id % total_vertices);

                    for(types::Position position = 0; position < config::walk_length; position++)
                    {
                        if (!graph[state.first].samplers->contains(state.second))
                            graph[state.first].samplers->insert(state.second, MetropolisHastingsSampler(state, model));

                        auto new_state = graph[state.first].samplers->find(state.second).sample(state, model);
                        new_state = model->new_state(state, graph[state.first].neighbors[random.irand(graph[state.first].degree)]);

                        // todo: check the neighbours here
                        if (state.first == new_state.first)
                        {
                            cout << "MYMY - wid=" << walk_id
                            << ", next_vertex=" << new_state.first
                            << " current_vertex=" << state.first
                            << "with triplet to encode {wid=" << walk_id << ", pos=" << (int) position << ", nxt=" << new_state.first << "}" << endl;
                        }

                        if (!cuckoo.contains(state.first))
                            cuckoo.insert(state.first, std::vector<types::Vertex>());

                        types::PairedTriplet hash = (position != config::walk_length - 1) ?
                                pairings::Szudzik<types::Vertex>::pair({walk_id * config::walk_length + position, new_state.first}) :
                                pairings::Szudzik<types::Vertex>::pair({walk_id * config::walk_length + position, state.first}); // assign the current as next if EOW
//                                pairings::Szudzik<types::Vertex>::pair({walk_id * config::walk_length + position,std::numeric_limits<uint32_t>::max() - 1});

                        cuckoo.update_fn(state.first, [&](auto& vector) {
                            vector.push_back(hash);
                        });

                        // ----------------------------------------------------------------------
                        // Refine the next (min, max) for the current vertex -------------------- // todo: can make the code more succint
                        if (position != config::walk_length - 1)
                        {
                            next_min[state.first] = std::min(next_min[state.first], new_state.first);
                            next_max[state.first] = std::max(next_max[state.first], new_state.first);
//                            cout << "*** next in the walk, min=" << next_min[state.first] << " and max=" << next_max[state.first] << endl;
                        }
                        else
                        {
                            next_min[state.first] = std::min(next_min[state.first], state.first);
                            next_max[state.first] = std::max(next_max[state.first], state.first);
                            //                            cout << "*** next in the walk, min=" << next_min[state.first] << " and max=" << next_max[state.first] << endl;
                        }
                        // ---------------------------------------------------------------

                        // Assign the new state to the sampler
                        state = new_state;
                    }
                });

//                #ifdef MALIN_DEBUG
//                for (auto i = 0; i < this->number_of_vertices(); i++)
//                {
//                    cout << "vertex=" << i << " has [lb=" << next_min[i] << ", ub=" << next_max[i] << "]" << endl;
//                    assert(next_min[i] <= next_max[i]); // lb should always be less than or equal than the upper bound
//                }
//                #endif

                // 2. build vertices
                parallel_for(0, total_vertices, [&](types::Vertex vertex)
                {
                    if (cuckoo.contains(vertex))
                    {
                        auto triplets = cuckoo.find(vertex);
                        auto sequence = pbbs::sequence<types::Vertex>(triplets.size());

                        for(auto index = 0; index < triplets.size(); index++)
                            sequence[index] = triplets[index];

                        pbbs::sample_sort_inplace(pbbs::make_range(sequence.begin(), sequence.end()), std::less<>());
                        vertices[vertex] = std::make_pair(vertex, VertexEntry(types::CompressedEdges(), dygrl::CompressedWalks(sequence, vertex, next_min[vertex], next_max[vertex]),new dygrl::SamplerManager(0)));
                    }
                    else
                    {
                        vertices[vertex] = std::make_pair(vertex, VertexEntry(types::CompressedEdges(), dygrl::CompressedWalks(),new dygrl::SamplerManager(0)));
                    }
                });

//                for(auto& e : vertices)
//                {
//                    std::cout << e.first << " ";
//                }
//
//                std::cout << std::endl;

                auto replace = [&] (const uintV& src, const VertexEntry& x, const VertexEntry& y)
                {
                    auto tree_plus = walk_plus::uniont(x.compressed_walks, y.compressed_walks, src);

                    // deallocate the memory
                    lists::deallocate(x.compressed_walks.plus);
                    walk_plus::Tree_GC::decrement_recursive(x.compressed_walks.root);
                    lists::deallocate(y.compressed_walks.plus);
                    walk_plus::Tree_GC::decrement_recursive(y.compressed_walks.root);

                    return VertexEntry(x.compressed_edges,
                                       CompressedWalks(tree_plus.plus, tree_plus.root, y.compressed_walks.vnext_min, y.compressed_walks.vnext_max),
                                       x.sampler_manager);
                };

                this->graph_tree = Graph::Tree::multi_insert_sorted_with_values(this->graph_tree.root, vertices.begin(), vertices.size(), replace, true);
                delete model;
            }

            /**
             * @brief Walks through the walk given walk id.
             *
             * @param walk_id - unique walk ID
             *
             * @return - walk string representation
             */
            std::string walk(types::WalkID walk_id)
            {
                // 1. Grab the first vertex in the walk
                types::Vertex current_vertex = walk_id % this->number_of_vertices();
                std::stringstream string_stream;
                types::Position position = 0;

                // 2. Walk
                types::Vertex previous_vertex = -1;
//                while (current_vertex != std::numeric_limits<uint32_t>::max() - 1)
                while (previous_vertex != current_vertex)
//                while (true)
                {
                    string_stream << current_vertex << " ";

                    auto tree_node = this->graph_tree.find(current_vertex);

                    #ifdef MALIN_DEBUG
                        if (!tree_node.valid)
                        {
                            std::cerr << "Malin debug error! Malin::Walk::Vertex="
                                      << current_vertex << " is not found in the vertex tree!"
                                      << std::endl;

                            std::exit(1);
                        }
                    #endif

                    // Cache the previous current vertex before going to the next
                    previous_vertex = current_vertex;

                    if (config::range_search_mode)
                        current_vertex = tree_node.value.compressed_walks.find_next_in_range(walk_id, position++, current_vertex);
                    else
                        current_vertex = tree_node.value.compressed_walks.find_next(walk_id, position++, current_vertex);

//                    if (current_vertex == previous_vertex)
//                        break; //Assumption: The datasets do not have self-loops.
                }

                return string_stream.str();
            }

            /**
             * @brief Walks through the walk given walk id.
             * todo: remove this function in the end
             * @param walk_id - unique walk ID
             *
             * @return - walk string representation
             */
            std::string walk_simple_find(types::WalkID walk_id)
            {
                // 1. Grab the first vertex in the walk
                types::Vertex current_vertex = walk_id % this->number_of_vertices();
                std::stringstream string_stream;
                types::Position position = 0;

                // 2. Walk
                types::Vertex previous_vertex;
//                while (current_vertex != std::numeric_limits<uint32_t>::max() - 1)
                while (true)
                {
                    string_stream << current_vertex << " ";

                    auto tree_node = this->graph_tree.find(current_vertex);

                    #ifdef MALIN_DEBUG
                        if (!tree_node.valid)
                        {
                            std::cerr << "Malin debug error! Malin::Walk::Vertex="
                                      << current_vertex << " is not found in the vertex tree!"
                                      << std::endl;

                            std::exit(1);
                        }
                    #endif

                    // Cache the previous current vertex before going to the next
                    previous_vertex = current_vertex;

                    // uses only simple find_next
                    current_vertex = tree_node.value.compressed_walks.find_next(walk_id, position++, current_vertex);

                    if (current_vertex == previous_vertex)
                        break;
                }

                return string_stream.str();
            }

            // todo: left to right traversal to find the previous vertex in a walk (highly inefficient)
            // todo: (Solution 1) extend wharf to support right to left traversal too
            // todo: (Solution 2) cache somehow in each node of a walk the previous node id as well (tailored to 2nd order walks)
            types::Vertex vertex_at_walk(types::WalkID walk_id, types::Position position)
            {
                types::Vertex current_vertex = walk_id % this->number_of_vertices();

                for (types::Position pos = 0; pos < position; pos++)
                {
                    auto tree_node = this->graph_tree.find(current_vertex);

                    #ifdef MALIN_DEBUG
                        if (!tree_node.valid)
                        {
                            std::cerr << "Malin debug error! Malin::Walk::Vertex="
                                      << current_vertex << " is not found in the vertex tree!"
                                      << std::endl;

                            std::exit(1);
                        }
                    #endif

                    if (config::range_search_mode)
                        current_vertex = tree_node.value.compressed_walks.find_next_in_range(walk_id, pos, current_vertex);
                    else
                        current_vertex = tree_node.value.compressed_walks.find_next(walk_id, pos, current_vertex);
                }

                return current_vertex;
            }

            /**
            * @brief Inserts a batch of edges in the graph.
            *
            * @param m                  - size of the batch
            * @param edges              - atch of edges to insert
            * @param sorted             - sort the edges in the batch
            * @param remove_dups        - removes duplicate edges in the batch
            * @param nn
            * @param apply_walk_updates - decides if walk updates will be executed
            */
            pbbs::sequence<types::WalkID> insert_edges_batch(size_t m,
                                                             std::tuple<uintV, uintV>* edges,
                                                             bool sorted = false,
                                                             bool remove_dups = false,
                                                             size_t nn = std::numeric_limits<size_t>::max(),
                                                             bool apply_walk_updates = true,
                                                             bool run_seq = false)
            {
                auto fl = run_seq ? pbbs::fl_sequential : pbbs::no_flag;

                // 1. Set up
                using Edge = std::tuple<uintV, uintV>;

                auto edges_original = pbbs::make_range(edges, edges + m);
                Edge* edges_deduped = nullptr;

                // 2. Sort the edges in the batch (by source)
                if (!sorted)
                {
                    Malin::sort_edge_batch_by_source(edges, m, nn);
                }

                // 3. Remove duplicate edges
                if (remove_dups)
                {
                    // true when no duplicated edge, false otherwise
                    auto bool_seq = pbbs::delayed_seq<bool>(edges_original.size(), [&] (size_t i)
                    {
                        if(get<0>(edges_original[i]) == get<1>(edges_original[i])) return false;
                        return (i == 0 || edges_original[i] != edges_original[i-1]);
                    });

                    auto E = pbbs::pack(edges_original, bool_seq, fl); // Creates a new pbbs::sequence
                    auto m_initial = m;                                // Initial number of generated edges
                    m = E.size();                                      // The size is not the same
                    auto m_final = m;                                  // Final number of edges in batch after duplicate removal
                    edges_deduped = E.to_array();                      // E afterwards is empty and nullptr
                }

                auto E = (edges_deduped) ? pbbs::make_range(edges_deduped, edges_deduped + m) : edges_original;

                // 4. Pack the starts vertices of edges
                auto start_im = pbbs::delayed_seq<size_t>(m, [&] (size_t i)
                {
                    return (i == 0 || (get<0>(E[i]) != get<0>(E[i-1])));
                });

                auto starts = pbbs::pack_index<size_t>(start_im, fl);
                size_t num_starts = starts.size();

                // 5. Build new wharf vertices
                using KV = std::pair<uintV, VertexEntry>;

                // Decides to store Wharf vertices on stack or heap
                constexpr const size_t stack_size = 20;
                KV kv_stack[stack_size];
                KV* new_verts = kv_stack;
                if (num_starts > stack_size)
                    new_verts = pbbs::new_array<KV>(num_starts);

                // pack the edges in the form: vertex_id - array of new edges
                parallel_for(0, num_starts, [&] (size_t i) {
                    size_t off = starts[i];
                    size_t deg = ((i == (num_starts-1)) ? m : starts[i+1]) - off;
                    uintV v = get<0>(E[starts[i]]);

                    auto S = pbbs::delayed_seq<uintV>(deg, [&] (size_t i) { return get<1>(E[off + i]); });

                    new_verts[i] = make_pair(v, VertexEntry(types::CompressedEdges(S, v, fl), dygrl::CompressedWalks(), new dygrl::SamplerManager(0)));
                });

                types::MapOfChanges rewalk_points = types::MapOfChanges();

                auto replace = [&, run_seq] (const intV& v, const VertexEntry& a, const VertexEntry& b)
                {
                    auto union_edge_tree = edge_plus::uniont(b.compressed_edges, a.compressed_edges, v, run_seq);

                    lists::deallocate(a.compressed_edges.plus);
                    edge_plus::Tree_GC::decrement_recursive(a.compressed_edges.root, run_seq);

                    lists::deallocate(b.compressed_edges.plus);
                    edge_plus::Tree_GC::decrement_recursive(b.compressed_edges.root, run_seq);

                    a.compressed_walks.iter_elms(v, [&](auto value)
                    {
                        auto pair = pairings::Szudzik<types::PairedTriplet>::unpair(value);

                        auto walk_id = pair.first / config::walk_length;
                        auto position = pair.first - (walk_id * config::walk_length);
                        auto next = pair.second;

                        if (!rewalk_points.template contains(walk_id))
                        {
                            rewalk_points.template insert(walk_id, std::make_tuple(position, v, false));
                        }
                        else
                        {
                            types::Position current_min_pos = get<0>(rewalk_points.find(walk_id));

                            if (current_min_pos > position)
                            {
                                rewalk_points.template update(walk_id, std::make_tuple(position, v, false));
                            }
                        }
                    });

                    return VertexEntry(union_edge_tree, a.compressed_walks, b.sampler_manager); // todo: check this sampler manager!
                };

                graph_update_time_on_insert.start();
                this->graph_tree = Graph::Tree::multi_insert_sorted_with_values(this->graph_tree.root, new_verts, num_starts, replace,true, run_seq);
                graph_update_time_on_insert.stop();

                walk_update_time_on_insert.start();
                auto affected_walks = pbbs::sequence<types::WalkID>(rewalk_points.size());
                if (apply_walk_updates)
                {
//                    if (config::mini_batch_mode)
//                        this->mini_batch_walk_update(rewalk_points, affected_walks);
//                    else
                        this->batch_walk_update(rewalk_points, affected_walks);
                }
                walk_update_time_on_insert.stop();

                // 6. Deallocate memory
                if (num_starts > stack_size) pbbs::free_array(new_verts);
                if (edges_deduped)           pbbs::free_array(edges_deduped);

                #ifdef MALIN_DEBUG
                    std::cout << "Rewalk points (MapOfChanges): " << std::endl;

                    auto table = rewalk_points.lock_table();

                    for(auto& item : table)
                    {
                        std::cout << "Walk ID: " << item.first
                                  << " Position: "
                                  << (int) std::get<0>(item.second)
                                  << " Vertex: "
                                  << std::get<1>(item.second)
                                  << " Should reset: "
                                  << std::get<2>(item.second)
                                  << std::endl;
                    }

                    table.unlock();
                #endif

                return affected_walks;
            }

            /**
            * @brief Deletes a batch of edges from the graph.
            *
            * @param m - size of the batch
            * @param edges - batch of edges to delete
            * @param sorted - sort the edges in the batch
            * @param remove_dups - removes duplicate edges in the batch
            * @param nn
            * @param run_seq - decides if walk updates will be executed
            */
            pbbs::sequence<types::WalkID> delete_edges_batch(size_t m,
                                                             tuple<uintV,
                                                             uintV>* edges,
                                                             bool sorted = false,
                                                             bool remove_dups = false,
                                                             size_t nn = std::numeric_limits<size_t>::max(),
                                                             bool apply_walk_updates = true,
                                                             bool run_seq = false)
            {
                auto fl = run_seq ? pbbs::fl_sequential : pbbs::no_flag;

                // 1. Set up
                using Edge = tuple<uintV, uintV>;

                auto edges_original = pbbs::make_range(edges, edges + m);
                Edge* edges_deduped = nullptr;

                // 2. Sort the edges in the batch (by source)
                if (!sorted)
                {
                    Malin::sort_edge_batch_by_source(edges, m, nn);
                }

                // 3. Remove duplicate edges
                if (remove_dups)
                {
                    // true when no duplicated edge, false otherwise
                    auto bool_seq = pbbs::delayed_seq<bool>(edges_original.size(), [&] (size_t i)
                    {
                        if(get<0>(edges_original[i]) == get<1>(edges_original[i])) return false;
                        return (i == 0 || edges_original[i] != edges_original[i-1]);
                    });

                    auto E = pbbs::pack(edges_original, bool_seq, fl); // creates a new pbbs::sequence
                    auto m_initial = m;                                // Initial number of generated edges
                    m = E.size();                                      // the size is not the same
                    auto m_final = m;                                  // Final number of edges in batch after duplicate removal
                    edges_deduped = E.to_array();                      // E afterwards is empty and nullptr
                }

                auto E = (edges_deduped) ? pbbs::make_range(edges_deduped, edges_deduped + m) : edges_original;

                // 4. Pack the starts vertices of edges
                auto start_im = pbbs::delayed_seq<size_t>(m, [&] (size_t i)
                {
                    return (i == 0 || (get<0>(E[i]) != get<0>(E[i-1])));
                });

                auto starts = pbbs::pack_index<size_t>(start_im, fl);
                size_t num_starts = starts.size();

                // 5. Build new wharf vertices
                using KV = std::pair<uintV, VertexEntry>;

                // decides to store whatf vertices on stack or heap
                constexpr const size_t stack_size = 20;
                KV kv_stack[stack_size];
                KV* new_verts = kv_stack;
                if (num_starts > stack_size)
                {
                    new_verts = pbbs::new_array<KV>(num_starts);
                }

                // pack the edges in the form: vertex_id - array of new edges
                parallel_for(0, num_starts, [&] (size_t i) {
                    size_t off = starts[i];
                    size_t deg = ((i == (num_starts-1)) ? m : starts[i+1]) - off;
                    uintV v = get<0>(E[starts[i]]);

                    auto S = pbbs::delayed_seq<uintV>(deg, [&] (size_t i) { return get<1>(E[off + i]); });

                    new_verts[i] = make_pair(v, VertexEntry(types::CompressedEdges(S, v, fl), dygrl::CompressedWalks(), new SamplerManager(0)));
                });

                types::MapOfChanges rewalk_points = types::MapOfChanges();

                auto replace = [&,run_seq] (const intV& v, const VertexEntry& a, const VertexEntry& b)
                {
                    auto difference_edge_tree = edge_plus::difference(b.compressed_edges, a.compressed_edges, v, run_seq);

                    lists::deallocate(a.compressed_edges.plus);
                    edge_plus::Tree_GC::decrement_recursive(a.compressed_edges.root, run_seq);

                    lists::deallocate(b.compressed_edges.plus);
                    edge_plus::Tree_GC::decrement_recursive(b.compressed_edges.root, run_seq);

                    bool should_reset = difference_edge_tree.degree() == 0;

                    a.compressed_walks.iter_elms(v, [&](auto value)
                    {
                        auto pair = pairings::Szudzik<types::PairedTriplet>::unpair(value);

                        auto walk_id = pair.first / config::walk_length;
                        auto position = pair.first - (walk_id * config::walk_length);
                        auto next = pair.second;

                        if (!rewalk_points.template contains(walk_id))
                        {
                            rewalk_points.template insert(walk_id, std::make_tuple(position, v, should_reset));
                        }
                        else
                        {
                            types::Position current_min_pos = get<0>(rewalk_points.find(walk_id));

                            if (current_min_pos > position)
                            {
                                rewalk_points.template update(walk_id, std::make_tuple(position, v, should_reset));
                            }
                        }
                    });

                    return VertexEntry(difference_edge_tree, a.compressed_walks, b.sampler_manager);
                };

                graph_update_time_on_delete.start();
                this->graph_tree = Graph::Tree::multi_insert_sorted_with_values(this->graph_tree.root, new_verts, num_starts, replace, true, run_seq);
                graph_update_time_on_delete.stop();

                walk_update_time_on_delete.start();
                auto affected_walks = pbbs::sequence<types::WalkID>(rewalk_points.size());
                if (apply_walk_updates)
                {
//                    if (config::mini_batch_mode)
//                        this->mini_batch_walk_update(rewalk_points, affected_walks);
//                    else
                        this->batch_walk_update(rewalk_points, affected_walks);
                }
                walk_update_time_on_delete.stop();

                // 6. Deallocate memory
                if (num_starts > stack_size) pbbs::free_array(new_verts);
                if (edges_deduped) pbbs::free_array(edges_deduped);

                #ifdef MALIN_DEBUG
                    std::cout << "Rewalk points (MapOfChanges): " << std::endl;

                    auto table = rewalk_points.lock_table();

                    for(auto& item : table)
                    {
                        std::cout << "Walk ID: " << item.first
                                  << " Position: "
                                  << (int) std::get<0>(item.second)
                                  << " Vertex: "
                                  << std::get<1>(item.second)
                                  << " Should reset: "
                                  << std::get<2>(item.second)
                                  << std::endl;
                    }

                    table.unlock();
                #endif

                return affected_walks;
            }

            /**
             * @brief Updates affected walks in batch mode. (VERSION 1)
             * // todo: Compare this initial version of batch walk updates
             * @param types::MapOfChanges - rewalking points
             */
            void batch_walk_update(types::MapOfChanges& rewalk_points, pbbs::sequence<types::WalkID>& affected_walks)
            {
                types::ChangeAccumulator deletes = types::ChangeAccumulator();
                types::ChangeAccumulator inserts = types::ChangeAccumulator();

                // --------------  Range Search Code Part  --------------------------------------------------
                constexpr const size_t array_next_size = 20;
                types::Vertex next_min_D_stack[array_next_size], next_max_D_stack[array_next_size], next_min_I_stack[array_next_size], next_max_I_stack[array_next_size];
                types::Vertex* next_min_wtree_D = next_min_D_stack;
                types::Vertex* next_max_wtree_D = next_max_D_stack;
                types::Vertex* next_min_wtree_I = next_min_I_stack;
                types::Vertex* next_max_wtree_I = next_max_I_stack;
                if (this->number_of_vertices() > array_next_size)
                {
                    next_min_wtree_D = pbbs::new_array<types::Vertex>(this->number_of_vertices());
                    next_max_wtree_D = pbbs::new_array<types::Vertex>(this->number_of_vertices());
                    next_min_wtree_I = pbbs::new_array<types::Vertex>(this->number_of_vertices());
                    next_max_wtree_I = pbbs::new_array<types::Vertex>(this->number_of_vertices());
                }
                parallel_for(0, this->number_of_vertices(), [&] (types::Vertex i) {
                    next_min_wtree_D[i] = std::numeric_limits<uint32_t>::max();
                    next_max_wtree_D[i] = 0;
                    next_min_wtree_I[i] = std::numeric_limits<uint32_t>::max();
                    next_max_wtree_I[i] = 0;
                });
                // -----------------------------------------------------------------------------------------

                uintV index = 0;

                for(auto& entry : rewalk_points.lock_table()) // todo: blocking?
                {
                    affected_walks[index++] = entry.first;
                }

                auto graph = this->flatten_graph();
                RandomWalkModel* model;

                switch (config::random_walk_model)
                {
                    case types::DEEPWALK:
                        model = new DeepWalk(&graph);
                        break;
                    case types::NODE2VEC:
                        model = new Node2Vec(&graph, config::paramP, config::paramQ);
                        break;
                    default:
                        std::cerr << "Unrecognized random walking model!" << std::endl;
                        std::exit(1);
                }

                // Parallel Update of Affected Walks
                parallel_for(0, affected_walks.size(), [&](auto index)
                {
                    auto entry = rewalk_points.template find(affected_walks[index]);

                    auto current_position        = std::get<0>(entry);
                    auto current_vertex_old_walk = std::get<1>(entry);
                    auto should_reset            = std::get<2>(entry);
//                    cout << "--- worker-" << worker_id() << " operates on walk-" << index << " from pos=" << (int) current_position << " and vertex=" << current_vertex_old_walk << endl;

                    auto current_vertex_new_walk = current_vertex_old_walk;

                    if (should_reset) // todo: clear out if this is needed
                    {
                        current_position = 0;
                        current_vertex_new_walk = current_vertex_old_walk = affected_walks[index] % this->number_of_vertices();
                    }

                    fork_join_scheduler::Job insert_job = [&] ()
                    {
                        if (graph[current_vertex_new_walk].degree == 0)
                        {
//                            types::PairedTriplet hash = pairings::Szudzik<types::Vertex>::pair({affected_walks[index]*config::walk_length, std::numeric_limits<uint32_t>::max() - 1});
                            types::PairedTriplet hash = pairings::Szudzik<types::Vertex>::pair({affected_walks[index] * config::walk_length + 0, current_vertex_new_walk});

                            if (!inserts.contains(current_vertex_new_walk)) inserts.insert(current_vertex_new_walk, std::vector<types::PairedTriplet>());
                            inserts.update_fn(current_vertex_new_walk, [&](auto& vector) {
                                vector.push_back(hash);
                            });

                            // ------------------------- Search In Range Code -----------------------------------------------------------------------
                            // Refine the (min, max) I for the walk-tree of current_vertex_new_walk -------------------------------------------------
                            next_min_wtree_I[current_vertex_new_walk] = std::min(next_min_wtree_I[current_vertex_new_walk], current_vertex_new_walk);
                            next_max_wtree_I[current_vertex_new_walk] = std::max(next_max_wtree_I[current_vertex_new_walk], current_vertex_new_walk);
                            // ----------------------------------------------------------------------------------------------------------------------

                            return;
                        }

                        // todo: this creates deterministic walk updates. Parameterize this in globals along with the other appearances
                        auto random = utility::Random(affected_walks[index] / this->number_of_vertices()); // todo: comment this line out when no determinism
                        auto state = model->initial_state(current_vertex_new_walk);

                        if (config::random_walk_model == types::NODE2VEC && current_position > 0)
                        {
                            state.first  = current_vertex_new_walk;
                            state.second = this->vertex_at_walk(affected_walks[index], current_position - 1); // todo: inefficient
                        }

                        for (types::Position position = current_position; position < config::walk_length; position++)
                        {
                            if (!graph[state.first].samplers->contains(state.second))
                                graph[state.first].samplers->insert(state.second, MetropolisHastingsSampler(state, model));

                            auto cached_current_vertex = state.first; // important for the correct access of the graph vertex
                            state = graph[state.first].samplers->find(state.second).sample(state, model);
//                            state = model->new_state(state, graph[state.first].neighbors[random.irand(graph[state.first].degree)]);
                            state = model->new_state(state, graph[cached_current_vertex].neighbors[random.irand(graph[cached_current_vertex].degree)]);

////                            cout << "worker-" << worker_id() << " wid-" << affected_walks[index] << " current-" << current_vertex_new_walk << " next-" << state.first << endl;
//                            // todo: check the neighbours here
//                            if (state.first == current_vertex_new_walk)
//                            {
//                                // ----------------------------------------------------------
//                                cout << "vertex=" << cached_current_vertex << "-- ";
//                                for(auto i = 0; i < graph[cached_current_vertex].degree; i++)
//                                    cout << " " << graph[cached_current_vertex].neighbors[i];
//                                cout << endl;
//                                // ----------------------------------------------------------
//
//                                cout << "HEY - wid=" << affected_walks[index]
//                                     << ", state.first=" << state.first
//                                     << " cached_current_vertex=" << current_vertex_new_walk
//                                     << " with triplet to encode {wid=" << affected_walks[index] << ", pos=" << (int) position << ", nxt=" << state.first << "}" << endl;
//                            }
//                            assert(state.first != current_vertex_new_walk);
////                            // --------------------------------------------

                            types::PairedTriplet hash = (position != config::walk_length - 1) ?
                                pairings::Szudzik<types::Vertex>::pair({affected_walks[index] * config::walk_length + position, state.first}) : // new sampled next
                                pairings::Szudzik<types::Vertex>::pair({affected_walks[index] * config::walk_length + position, current_vertex_new_walk});
//                                pairings::Szudzik<types::Vertex>::pair({affected_walks[index]*config::walk_length + position, std::numeric_limits<uint32_t>::max() - 1});
//                            cout << "Insertion wid=" << index << ", pos=" << (int) position << ", next=" << ((position != config::walk_length - 1) ? state.first : cached_current_vertex) << " ===> pairedTriplet=" << hash << endl;

                            if (!inserts.contains(current_vertex_new_walk)) inserts.insert(current_vertex_new_walk, std::vector<types::PairedTriplet>());
                            inserts.update_fn(current_vertex_new_walk, [&](auto& vector) {
                                vector.push_back(hash);
                            });

                            // ------------------------- Search In Range Code ----- todo: ensure correctness!
                            // Refine the (min, max) I for the walk-tree of current_vertex_new_walk
                            if (position != config::walk_length - 1)
                            {
                                next_min_wtree_I[current_vertex_new_walk] = std::min(next_min_wtree_I[current_vertex_new_walk], state.first);
                                next_max_wtree_I[current_vertex_new_walk] = std::max(next_max_wtree_I[current_vertex_new_walk], state.first);
                            }
                            else
                            {
                                next_min_wtree_I[current_vertex_new_walk] = std::min(next_min_wtree_I[current_vertex_new_walk], current_vertex_new_walk);
                                next_max_wtree_I[current_vertex_new_walk] = std::max(next_max_wtree_I[current_vertex_new_walk], current_vertex_new_walk);
//                                cout << "NEW wid=" << affected_walks[index] << " inserted last paired triplet" << endl;
                            }
                            // -------------------------------------------------------------------------

                            // Then, change the current vertex in the new walk
                            current_vertex_new_walk = state.first;
                        }

                    }; insert_job(); //todo: figure out how to actually push the job to the scheduler

                    fork_join_scheduler::Job delete_job = [&] ()
                    {
                        types::Position position = current_position;

                        types::Vertex cached_current_vertex = -1;
//                        while (current_vertex_old_walk != std::numeric_limits<uint32_t>::max() - 1)
                        while (current_vertex_old_walk != cached_current_vertex)
                        {
                            auto vertex = this->graph_tree.find(current_vertex_old_walk);

                            types::Vertex next_old_walk;
                            if (config::range_search_mode)
                                next_old_walk = vertex.value.compressed_walks.find_next_in_range(affected_walks[index], position, current_vertex_old_walk);
                            else
                                next_old_walk = vertex.value.compressed_walks.find_next(affected_walks[index], position, current_vertex_old_walk);

                            types::PairedTriplet hash = pairings::Szudzik<types::Vertex>::pair({affected_walks[index] * config::walk_length + position, next_old_walk});
//                            cout << "Deletion wid=" << index << ", pos=" << (int) position << ", next=" << next_old_walk << " ===> pairedTriplet=" << hash << endl;

                            if (!deletes.contains(current_vertex_old_walk)) deletes.insert(current_vertex_old_walk, std::vector<types::PairedTriplet>());
                            deletes.update_fn(current_vertex_old_walk, [&](auto& vector) {
                                vector.push_back(hash);
                            });

                            // ---- Search in Range Code ---- todo: This may be skipped as (min_D, max_D) subset of (min_CW, max_CW)
                            // Refine the (min, max) for node current_vertex_old_walk
                            next_min_wtree_D[current_vertex_old_walk] = std::min(next_min_wtree_D[current_vertex_old_walk], next_old_walk);
                            next_max_wtree_D[current_vertex_old_walk] = std::max(next_min_wtree_D[current_vertex_old_walk], next_old_walk);
                            // -------------------------------------------------------------------------

                            // break the loop, after inserting the last triplet to the deletion accumulator
//                            if (current_vertex_old_walk == next_old_walk)
//                                break;

                            // Then, transition to the next vertex in the old walk
                            position++;
                            cached_current_vertex   = current_vertex_old_walk; // cache the current vertex
                            current_vertex_old_walk = next_old_walk;

//                            if (cached_current_vertex == current_vertex_old_walk)
//                                cout << "OLD wid=" << affected_walks[index] << " deleted last paired triplet" << endl;
                        }
                    }; delete_job();

                });



                // --------------------------------------------------------- todo: debug the freaking accumulator production
                // print out the ranges in the delete and insert batches
//                for (auto i = 0; i < this->number_of_vertices(); i++)
//                {
//                    cout << "vertex " << i << ", next I_min=" << next_min_wtree_I[i] << ", I_max=" << next_max_wtree_I[i] << endl;
//                    cout << "vertex " << i << ", next D_min=" << next_min_wtree_I[i] << ", D_max=" << next_max_wtree_D[i] << endl;
//                }

                // --------- For debugging ---------------------- ---------------------------------------------------
//                auto totalItriplets = 0;
//                auto totalDtriplets = 0;
//                auto inserts_vertices = pbbs::sequence<types::Vertex>(inserts.size());
//                for (auto& item : inserts.lock_table())
//                    totalItriplets += item.second.size();
//                auto deletes_vertices = pbbs::sequence<types::Vertex>(deletes.size());
//                for (auto& item : deletes.lock_table())
//                    totalDtriplets += item.second.size();
//                cout << "Total #tripletsD=" << totalDtriplets << " and total #tripletsI=" << totalItriplets << endl;
//                assert(totalDtriplets == totalItriplets); // only after inserting a batch this holds. when deleting edges this might not hold
                // For instance, a vertex is connected with one edge with the rest of the graph and after deletion of this edge then the walk
                // that initiates from it has only one element.
                // --------------------------------------------------------------------------------------------------

                using VertexStruct  = std::pair<types::Vertex, VertexEntry>;
                auto insert_walks  = pbbs::sequence<VertexStruct>(inserts.size());
                auto delete_walks  = pbbs::sequence<VertexStruct>(deletes.size());

                // fj.parfor or parallel_for
                fj.pardo([&]()
                {
                    auto ind = 0;

                    for(auto& item : inserts.lock_table()) // todo: pardo and lock_table?
                    {
                        auto sequence = pbbs::sequence<types::Vertex>(item.second.size());

                        for(auto i = 0; i < item.second.size(); i++)
                            sequence[i] = item.second[i];

                        pbbs::sample_sort_inplace(pbbs::make_range(sequence.begin(), sequence.end()), std::less<>());
                        insert_walks[ind++] = std::make_pair(item.first,VertexEntry(types::CompressedEdges(),
                            dygrl::CompressedWalks(sequence, item.first, next_min_wtree_I[item.first], next_max_wtree_I[item.first]),
                            new dygrl::SamplerManager(0)));
                    }
                },
                [&]()
                {
                    auto ind = 0;

                    for(auto& item : deletes.lock_table())
                    {
                        auto sequence = pbbs::sequence<types::Vertex>(item.second.size());

                        for (auto i = 0; i < item.second.size(); i++)
                            sequence[i] = item.second[i];

                        pbbs::sample_sort_inplace(pbbs::make_range(sequence.begin(), sequence.end()), std::less<>());
                        delete_walks[ind++] = std::make_pair(item.first, VertexEntry(types::CompressedEdges(),
                            dygrl::CompressedWalks(sequence, item.first, next_min_wtree_D[item.first], next_max_wtree_D[item.first]),
                            new dygrl::SamplerManager(0)));
                    }
                });

                fj.pardo([&]()
                {
                    pbbs::sample_sort_inplace(pbbs::make_range(delete_walks.begin(), delete_walks.end()), [&](auto& x, auto& y) { // todo: check make_range does not a copy
                        return x.first < y.first;
                    });
                }, [&]()
                {
                    pbbs::sample_sort_inplace(pbbs::make_range(insert_walks.begin(), insert_walks.end()), [&](auto& x, auto& y) {
                        return x.first < y.first;
                    });
                });

                auto replaceD = [&] (const uintV& src, const VertexEntry& x, const VertexEntry& y)
                {
                    auto tree_plus = walk_plus::difference(y.compressed_walks, x.compressed_walks, src); // x - y

                    // deallocate the memory
                    lists::deallocate(x.compressed_walks.plus);
                    walk_plus::Tree_GC::decrement_recursive(x.compressed_walks.root);
                    lists::deallocate(y.compressed_walks.plus);
                    walk_plus::Tree_GC::decrement_recursive(y.compressed_walks.root);

                    // --------- Search Range Code ------------------------------------------------------------
                    // Logic for ranges of next ids difference i.e., [x_min, x_max] + [y_min, y_max]
                    types::Vertex new_next_min = x.compressed_walks.vnext_min; // std::min(x.compressed_walks.vnext_min, y.compressed_walks.vnext_min);
                    types::Vertex new_next_max = x.compressed_walks.vnext_max; // std::max(x.compressed_walks.vnext_max, y.compressed_walks.vnext_max);
                    // ----------------------------------------------------------------------------------------

//                    cout << "vertex=" << src
//                    << " old walk-tree next range [lb=" << x.compressed_walks.vnext_min << ", ub=" << x.compressed_walks.vnext_max << "]"
//                    << " and next range of the edges to be deleted [lbD=" << y.compressed_walks.vnext_min << ", ubD=" << y.compressed_walks.vnext_max << "]"
//                    << " and final range is [lb'=" << new_next_min << ", ub'=" << new_next_max << "]" << endl;

                    return VertexEntry(x.compressed_edges, dygrl::CompressedWalks(tree_plus.plus, tree_plus.root, new_next_min, new_next_max), x.sampler_manager);
                };

                // First, apply the batch deletions
                this->graph_tree = Graph::Tree::multi_insert_sorted_with_values(this->graph_tree.root, delete_walks.begin(), delete_walks.size(), replaceD, true);

                auto replaceI = [&] (const uintV& src, const VertexEntry& x, const VertexEntry& y)
                {
                    auto tree_plus = walk_plus::uniont(x.compressed_walks, y.compressed_walks, src); // x + y

                    // deallocate the memory
                    lists::deallocate(x.compressed_walks.plus);
                    walk_plus::Tree_GC::decrement_recursive(x.compressed_walks.root);
                    lists::deallocate(y.compressed_walks.plus);
                    walk_plus::Tree_GC::decrement_recursive(y.compressed_walks.root);

                    // --------- Search Range Code ------------------------------------------------------------
                    // Logic for ranges of next ids difference i.e., [x_min, x_max] + [y_min, y_max]
                    types::Vertex new_next_min = std::min(x.compressed_walks.vnext_min, y.compressed_walks.vnext_min);
                    types::Vertex new_next_max = std::max(x.compressed_walks.vnext_max, y.compressed_walks.vnext_max);
                    // ----------------------------------------------------------------------------------------

//                    cout << "vertex=" << src
//                    << " old walk-tree next range [lb=" << x.compressed_walks.vnext_min << ", ub=" << x.compressed_walks.vnext_max << "]"
//                    << " and next range of the edges to be inserted [lbI=" << y.compressed_walks.vnext_min << ", ubI=" << y.compressed_walks.vnext_max << "]"
//                    << " and final range is [lb'=" << new_next_min << ", ub'=" << new_next_max << "]" << endl;

                    return VertexEntry(x.compressed_edges, dygrl::CompressedWalks(tree_plus.plus, tree_plus.root, new_next_min, new_next_max), x.sampler_manager);
                };

                // Then, apply the batch insertions
                this->graph_tree = Graph::Tree::multi_insert_sorted_with_values(this->graph_tree.root, insert_walks.begin(), insert_walks.size(), replaceI, true);

            } // End of batch walk update procedure



            /**
             * @brief Prints memory footprint details.
             */
            void memory_footprint() const
            {
                std::cout << std::endl;

                size_t graph_vertices   = this->number_of_vertices();
                size_t graph_edges      = this->number_of_edges();

                size_t vertex_node_size = Graph::node_size();
                size_t c_tree_node_size = types::CompressedTreesLists::node_size();

                size_t edges_heads      = 0;
                size_t walks_heads      = 0;
                size_t edges_bytes      = 0;
                size_t walks_bytes      = 0;
                size_t samplers_bytes   = 0;
                size_t flat_graph_bytes = 0;
                auto flat_graph         = this->flatten_vertex_tree();

                // measure size of {min,max} bounds for each walk-tree
                size_t vnext_min_bytes  = 0;
                size_t vnext_max_bytes  = 0;

                for (auto i = 0; i < flat_graph.size(); i++)
                {
                    flat_graph_bytes += sizeof(flat_graph[i]);

                    edges_heads += flat_graph[i].compressed_edges.edge_tree_nodes();
                    walks_heads += flat_graph[i].compressed_walks.edge_tree_nodes();

                    edges_bytes += flat_graph[i].compressed_edges.size_in_bytes(i);
                    walks_bytes += flat_graph[i].compressed_walks.size_in_bytes(i);

                    for (auto& entry : flat_graph[i].sampler_manager->lock_table())
                    {
                        samplers_bytes += sizeof(entry.first) + sizeof(entry.second);
                    }

                    // measure the size of all {min, max} bounds of range search
                    vnext_min_bytes   += sizeof(flat_graph[i].compressed_walks.vnext_min);
                    vnext_max_bytes   += sizeof(flat_graph[i].compressed_walks.vnext_max);
                }

                std::cout << "Graph: \n\t" << "Vertices: " << graph_vertices << ", Edges: " << graph_edges << std::endl;

                std::cout << "Vertex Tree: \n\t"
                          << "Heads: " << Graph::used_node()
                          << ", Head size: " << vertex_node_size
                          << ", Memory usage: " << utility::MB(Graph::get_used_bytes()) << " MB"
                          << " = " << utility::GB(Graph::get_used_bytes()) << " GB" << std::endl;

                std::cout << "Edge Trees: \n\t"
                          << "Heads: " << edges_heads
                          << ", Head size: " << c_tree_node_size
                          << ", Lists memory: " << utility::MB(edges_bytes) << " MB"
                          << " = " << utility::GB(edges_bytes) << " GB"
                          << ", Total memory usage: " << utility::MB(edges_bytes + edges_heads*c_tree_node_size)
                          << " MB = " << utility::GB(edges_bytes + edges_heads*c_tree_node_size)
                          << " GB" << std::endl;

                std::cout << "Walks Trees: \n\t"
                          << "Heads: " << walks_heads
                          << ", Head size: " << c_tree_node_size
                          << ", Lists memory: " << utility::MB(walks_bytes) << " MB"
                          << " = " << utility::GB(walks_bytes) << " GB"
                          << ", Total memory usage: " << utility::MB(walks_bytes + walks_heads*c_tree_node_size)
                          << " MB = " << utility::GB(walks_bytes + walks_heads*c_tree_node_size)
                          << " GB" << std::endl;

                // print out the size of bounds due to the range search
                std::cout << "Range Search (" << ((config::range_search_mode) ? "ON" : "OFF") << "): \n\t"
                          << "Total {min,max} memory usage: " << utility::MB(((config::range_search_mode) ? (vnext_min_bytes + vnext_max_bytes) : 0))
                          << " MB = " << utility::GB(((config::range_search_mode) ? (vnext_min_bytes + vnext_max_bytes) : 0))
                          << " GB" << std::endl;

                std::cout << "Samplers: \n\t"
                          << "Total memory usage: " << utility::MB(samplers_bytes)
                          << " MB = " << utility::GB(samplers_bytes)
                          << " GB" << std::endl;

                std::cout << "Flat graph: \n\t"
                          << "Total memory usage: " << utility::MB(flat_graph_bytes)
                          << " MB = " << utility::GB(flat_graph_bytes)
                          << " GB" << std::endl;

                size_t total_memory = Graph::get_used_bytes()
                        + walks_bytes + walks_heads*c_tree_node_size
                        + edges_bytes + edges_heads*c_tree_node_size + samplers_bytes + ((config::range_search_mode) ? (vnext_min_bytes + vnext_max_bytes) : 0);

                std::cout << "Total memory used: \n\t" << utility::MB(total_memory) << " MB = "
                          << utility::GB(total_memory) << " GB" << std::endl;

                std::cout << std::endl;
            }

            /**
             * @brief Prints memory pool stats for the underlying lists.
             */
            void print_memory_pool_stats() const
            {
                std::cout << std::endl;

                // vertices memory pool stats
                std::cout << "Vertices tree memory lists: \n\t";
                Graph::print_stats();

                // edges and walks memory pool stats
                std::cout << "Edges & Walks trees memory lists: \n\t";
                types::CompressedTreesLists::print_stats();

                // compressed lists
                std::cout << "Pluses & Tails memory lists: \n";
                compressed_lists::print_stats();

                std::cout << std::endl;
            }

        private:
            Graph graph_tree;

            /**
            * Initializes memory pools for underlying lists.
            *
            * uses default size init(): 1 000 000 blocks, each block is put into a list of size 65 536 blocks
            * total allocated blocks = list_size(=65 5336) * (thread_count(=1..n) + ceil(allocated_blocks(=1M) / list_size(=65 536))
            * maximum blocks = ((3*RAM)/4)/size of one block.
            * @see list_allocator.h
            *
            * @param graph_vertices - total graph vertices
            * @param graph_edges    - total graph edges
            */
            static void init_memory_pools(size_t graph_vertices, size_t graph_edges)
            {
                types::CompressedTreesLists::init();
                compressed_lists::init(graph_vertices);
                Graph::init();
            }

            /**
            * Sorts a sequence of batch updates.
            *
            * @param edges       sequence of edges (src, dst) to be sorted by src
            * @param batch_edges total number of edges to be sorted
            * @param nn
            */
            static void sort_edge_batch_by_source(std::tuple<uintV, uintV>* edges, size_t batch_edges, size_t nn = std::numeric_limits<size_t>::max())
            {
                #ifdef MALIN_TIMER
                    timer timer("Malin::SortEdgeBatchBySource");
                #endif

                // 1. Set up
                using Edge = tuple<uintV, uintV>;

                auto edges_original = pbbs::make_range(edges, edges + batch_edges);
                size_t vertex_bits = pbbs::log2_up(nn);     // number of bits to represent a vertex in the graph

                // 2. Induce nn if not given (captures x < number of nodes < y such that x and y are powers of 2)
                if (nn == std::numeric_limits<size_t>::max())
                {
                    auto max_edge_id = pbbs::delayed_seq<size_t>(batch_edges, [&](size_t i)
                    {
                        return std::max(std::get<0>(edges_original[i]), std::get<1>(edges_original[i]));
                    });

                    vertex_bits = pbbs::log2_up(pbbs::reduce(max_edge_id, pbbs::maxm<size_t>()));
                    nn = 1 << vertex_bits;
                }

                // 3. Sort edges by source
                auto edge_to_long = [nn, vertex_bits](Edge e) -> size_t {
                    return (static_cast<size_t>(std::get<0>(e)) << vertex_bits) + static_cast<size_t>(std::get<1>(e));
                };

    //                auto edge_ids_log = pbbs::delayed_seq<size_t>(batch_edges, [&](size_t i) {
    //                    return pbbs::log2_up(edge_to_long(edges_original[i]));
    //                });

                size_t bits = 2 * vertex_bits;

                // Only apply integer sort if it will be work-efficient
                if (nn <= (batch_edges * pbbs::log2_up(batch_edges)))
                {
                    pbbs::integer_sort_inplace(edges_original, edge_to_long, bits);
                }
                else
                {
                    pbbs::sample_sort_inplace(edges_original, std::less<>());
                }

                #ifdef MALIN_TIMER
                    timer.reportTotal("time (seconds)");
                #endif
            }
    };
}

#endif // DYNAMIC_GRAPH_REPRESENTATION_LEARNING_WITH_METROPOLIS_HASTINGS_MALIN_H
