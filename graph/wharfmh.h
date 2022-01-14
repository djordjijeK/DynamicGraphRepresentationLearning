#ifndef DYNAMIC_GRAPH_REPRESENTATION_LEARNING_WITH_METROPOLIS_HASTINGS_WharfMH_H
#define DYNAMIC_GRAPH_REPRESENTATION_LEARNING_WITH_METROPOLIS_HASTINGS_WharfMH_H

#include <graph/api.h>
#include <cuckoohash_map.hh>

#include <config.h>
#include <vertex.h>
#include <snapshot.h>

#include <models/deepwalk.h>
#include <models/node2vec.h>
#include <set>

namespace dynamic_graph_representation_learning_with_metropolis_hastings
{
    /**
     * @brief WharfMH represents a structure that stores a graph as an augmented parallel balanced binary tree.
     * Keys in this tree are graph vertices and values are compressed edges, and metropolis hastings samplers.
     */
    class WharfMH
    {
        public:
            using Graph       = aug_map<dygrl::Vertex>;
            using WalkStorage = libcuckoo::cuckoohash_map<types::WalkID, std::vector<types::Vertex>>;
			using WalkIndex   = libcuckoo::cuckoohash_map<types::Vertex, std::set<types::WalkID>>;

            /**
             * @brief WharfMH constructor.
             *
             * @param graph_vertices - total vertices in a graph
             * @param graph_edges    - total edges in a graph
             * @param offsets        - vertex offsets for its neighbors
             * @param edges          - edges
             * @param free_memory    - free memory excess after graph is loaded
             */
            WharfMH(long graph_vertices, long graph_edges, uintE* offsets, uintV* edges, bool free_memory = true)
            {
                #ifdef WHARFMH_TIMER
                    timer timer("WharfMH::Constructor");
                #endif

                // 1. Initialize memory pools
                WharfMH::init_memory_pools(graph_vertices, graph_edges);

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
                        vertices[index] = std::make_pair(index, VertexEntry(types::CompressedEdges(S, index), new dygrl::SamplerManager(0)));
                    else
                        vertices[index] = std::make_pair(index, VertexEntry(types::CompressedEdges(), new dygrl::SamplerManager(0)));
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

                #ifdef WHARFMH_TIMER
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
                #ifdef WHARFMH_TIMER
                    timer timer("WharfMH::FlattenVertexTree");
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

                #ifdef WHARFMH_TIMER
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
                #ifdef WHARFMH_TIMER
                    timer timer("WharfMH::FlattenGraph");
                #endif

                size_t n_vertices = this->number_of_vertices();
                auto flat_graph   = FlatGraph(n_vertices);

                auto map_func = [&] (const Graph::E& entry, size_t ind)
                {
                    const uintV& key  = entry.first;
                    const auto& value = entry.second;

                    flat_graph[key].neighbors = entry.second.compressed_edges.get_edges(key);
                    flat_graph[key].degrees   = entry.second.compressed_edges.degree();
                    flat_graph[key].samplers  = entry.second.sampler_manager;
                };

                this->map_vertices(map_func);

                #ifdef WHARFMH_TIMER
                    timer.reportTotal("time(seconds)");

                    auto size = flat_graph.size_in_bytes();

                    std::cout << "WharfMH::FlattenGraph: Flat graph memory footprint: "
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
             * @brief Destroys WharfMH instance.
             */
            void destroy()
            {
                this->graph_tree.~Graph();
                this->graph_tree.root = nullptr;
            }

            /**
            * @brief Destroys corpus of random walks.
            */
            void destroy_walks()
            {
                this->walk_storage.clear();
								// destroy the index too
								this->walk_index.clear();
            }

            /**
             * @brief Creates initial set of random walks.
             */
            void generate_initial_random_walks()
            {
                auto graph            = this->flatten_graph();
                auto total_vertices = this->number_of_vertices();
                auto walks          = total_vertices * config::walks_per_vertex;
//                auto cuckoo         = libcuckoo::cuckoohash_map<types::Vertex, std::vector<types::Vertex>>(total_vertices);

                using VertexStruct  = std::pair<types::Vertex, VertexEntry>;
                auto vertices       = pbbs::sequence<VertexStruct>(total_vertices);
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

				// initialize the inverted index. Create an entry for each nid ----------
				parallel_for(0, total_vertices, [&](types::Vertex nid)
	            {
                    this->walk_index.insert(nid, std::set<types::Vertex>()); // todo: check if this is correct
	            });
				// ----------------------------------------------------------------------

                parallel_for(0, walks, [&](types::WalkID walk_id)
                {
                    if (graph[walk_id % total_vertices].degrees == 0)
                    {
                        this->walk_storage.insert(walk_id, std::vector<types::Vertex>(config::walk_length));

                        this->walk_storage.update_fn(walk_id, [&](auto& vector)
                        {
                            vector.push_back(walk_id % total_vertices);
                        });

						// update the inverted index -----------------------------------------------------
						this->walk_index.insert(walk_id % total_vertices, std::set<types::Vertex>());
						this->walk_index.update_fn(walk_id % total_vertices,  [&](auto& set)
						{
								set.insert(walk_id);
						}); // todo
						// -------------------------------------------------------------------------------

                        return;
                    }

//                    this->walk_storage.insert(walk_id, std::vector<types::Vertex>(config::walk_length)); // do not initialize like this because it inserts zeros
					this->walk_storage.insert(walk_id, std::vector<types::Vertex>());


                    auto random = config::random;
					if (config::determinism)
						random = utility::Random(walk_id / total_vertices); // disable this if you need pure randomness
                    types::State state = model->initial_state(walk_id % total_vertices);

                    for(types::Position position = 0; position < config::walk_length; position++)
                    {
                        if (!graph[state.first].samplers->contains(state.second))
                        {
                            graph[state.first].samplers->insert(state.second, MetropolisHastingsSampler(state, model));
                        }

                        auto new_state = graph[state.first].samplers->find(state.second).sample(state, model);
	                    if (config::determinism)
                            new_state = model->new_state(state, graph[state.first].neighbors[random.irand(graph[state.first].degrees)]);

                        this->walk_storage.update_fn(walk_id, [&](auto& vector)
                        {
                            vector.push_back(state.first);
                        });

						// update the walk index as well --------------------------------
//						if (position == 0)
//								this->walk_index.insert(state.first, std::set<types::Vertex>()); // todo: check if this is correct
						this->walk_index.update_fn(state.first, [&](auto& set)
						{
								set.insert(walk_id);
						});
						// --------------------------------------------------------------

                        state = new_state;
                    }
                });

                delete model;
            }

			/**
			 * @brief Prints inverted index {nid, set(wid)}
			 *
			 * @param no parameters
			 *
			 * @return - inverted index for the walk corpus maintained
			 */
			 void walk_index_print()
			{
			    auto lt = this->walk_index.lock_table();
				for (const auto& it : lt)
				{
					cout << "nid: " << it.first << "- {";
					for (const auto& wid : it.second)
					{
						cout << wid << " ";
					}
					cout << "}" << endl;
				}
			}


            /**
             * @brief Walks through the walk given walk id.
             *
             * @param walk_id - unique walk ID
             *
             * @return - walk string representation
             */
            void walk_cout(types::WalkID walk_id)
            {
//                std::stringstream string_stream;
                if (!this->walk_storage.contains(walk_id))
                {
										cout << "No walk with id " << walk_id << endl;
										return;
								}

                for(auto& item : this->walk_storage.find(walk_id))
                {
//                    string_stream << item << " ";
										cout << item << " ";
                }
								cout << endl;

//                return string_stream.str();
								return;
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
                std::stringstream string_stream;
                if (!this->walk_storage.contains(walk_id)) return string_stream.str();

                for(auto& item : this->walk_storage.find(walk_id))
                {
                    string_stream << item << " ";
                }

                return string_stream.str();
            }

            /**
            * @brief Inserts a batch of edges in the graph.
            *
            * @param m - size of the batch
            * @param edges - atch of edges to insert
            * @param sorted - sort the edges in the batch
            * @param remove_dups - removes duplicate edges in the batch
            * @param nn
            * @param apply_walk_updates - decides if walk updates will be executed
            */
            int insert_edges_batch(size_t m,
								   std::tuple<uintV, uintV>* edges,
								   int batch_num,
								   bool sorted = false,
								   bool remove_dups = false,
								   size_t nn = std::numeric_limits<size_t>::max(),
								   bool apply_walk_updates = true,
								   bool run_seq = false)
            {
                #ifdef WHARFMH_TIMER
                    timer graph_update_time("WharfMH::InsertEdgesBatch::GraphUpdateTime");
                    timer walk_update_time("WharfMH::InsertEdgesBatch::WalkUpdateTime");
                #endif

                auto fl = run_seq ? pbbs::fl_sequential : pbbs::no_flag;

                // 1. Set up
                using Edge = std::tuple<uintV, uintV>;

                auto edges_original = pbbs::make_range(edges, edges + m);
                Edge* edges_deduped = nullptr;

                // 2. Sort the edges in the batch (by source)
                if (!sorted)
                {
                    WharfMH::sort_edge_batch_by_source(edges, m, nn);
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

                    new_verts[i] = make_pair(v, VertexEntry(types::CompressedEdges(S, v, fl), new dygrl::SamplerManager(0)));
                });

                types::MapOfChanges rewalk_points = types::MapOfChanges();
                auto replace = [&, run_seq] (const intV& v, const VertexEntry& a, const VertexEntry& b)
                {
                    auto union_edge_tree = tree_plus::uniont(b.compressed_edges, a.compressed_edges, v, run_seq);

                    lists::deallocate(a.compressed_edges.plus);
                    tree_plus::Tree_GC::decrement_recursive(a.compressed_edges.root, run_seq);

                    lists::deallocate(b.compressed_edges.plus);
                    tree_plus::Tree_GC::decrement_recursive(b.compressed_edges.root, run_seq);

					// --------------------------------------------------------------
					walk_update_time_on_insert.start();
					// use the inverted index somehow here
					if (this->walk_index.contains(v))
					{
						auto set_wids = this->walk_index.find(v);
//						cout << "set of " << v << " has " << set_wids.size() << " elements." << endl;

						// Go and visit only the wids in the inverted index to construct the MAV
						for (auto& wid : set_wids)
						{
							auto cur_walk = this->walk_storage.find(wid);
							for (types::Position position = 0; position < cur_walk.size(); position++)
							{
								if (cur_walk[position] != v) continue;

								if (!rewalk_points.contains(wid))
									rewalk_points.insert(wid, position);
								else
								{
									types::Position current_min_pos = rewalk_points.find(wid);

	                                if (current_min_pos > position)
	                                    rewalk_points.update(wid, position);
								}
							}
						}
					}

//                    for(auto& element : this->walk_storage.lock_table())
//                    {
//                        for (types::Position position = 0; position < element.second.size(); position++)
//                        {
//                            if (element.second[position] != v) continue;
//
//                            if (!rewalk_points.contains(element.first))
//                            {
//                                rewalk_points.insert(element.first, position);
//                            }
//                            else
//                            {
//                                types::Position current_min_pos = rewalk_points.find(element.first);
//
//                                if (current_min_pos > position)
//                                {
//                                    rewalk_points.update(element.first, position);
//                                }
//                            }
//                        }
//                    }
					walk_update_time_on_insert.stop();
					// -------------------------------------------------------------

                    return VertexEntry(union_edge_tree, a.sampler_manager);
                };

                #ifdef WHARFMH_TIMER
                    graph_update_time.start();
                #endif

                this->graph_tree = Graph::Tree::multi_insert_sorted_with_values(this->graph_tree.root, new_verts, num_starts, replace, true, run_seq);

                #ifdef WHARFMH_TIMER
                    graph_update_time.stop();
                #endif

                #ifdef WHARFMH_TIMER
                    walk_update_time.start();
                #endif

				walk_update_time_on_insert.start();
                if (apply_walk_updates) this->update_walks(rewalk_points);
				walk_update_time_on_insert.stop();

                #ifdef WHARFMH_TIMER
                    walk_update_time.stop();
                #endif

                // 6. Deallocate memory
                if (num_starts > stack_size) pbbs::free_array(new_verts);
                if (edges_deduped)           pbbs::free_array(edges_deduped);

                #ifdef WHARFMH_DEBUG
                    std::cout << "Rewalk points (MapOfChanges): " << rewalk_points.size() << std::endl;

                    for(auto& item : rewalk_points.lock_table())
                    {
                        std::cout << "Walk ID: " << item.first
                                  << " Position: "
                                  << (int) item.second
                                  << std::endl;
                    }
                #endif

                #ifdef WHARFMH_TIMER
                    graph_update_time.reportTotal("time(seconds)");
                    walk_update_time.reportTotal("time(seconds)");
                #endif

                return rewalk_points.size();
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
            int delete_edges_batch(size_t m, tuple<uintV, uintV>* edges, int batch_num, bool sorted = false, bool remove_dups = false, size_t nn = std::numeric_limits<size_t>::max(), bool apply_walk_updates = true, bool run_seq = false)
            {
                #ifdef WHARFMH_TIMER
                    timer graph_update_time("WharfMH::DeleteEdgesBatch::GraphUpdateTime");
                    timer walk_update_time("WharfMH::DeleteEdgesBatch::WalkUpdateTime");
                #endif

                auto fl = run_seq ? pbbs::fl_sequential : pbbs::no_flag;

                // 1. Set up
                using Edge = tuple<uintV, uintV>;

                auto edges_original = pbbs::make_range(edges, edges + m);
                Edge* edges_deduped = nullptr;

                // 2. Sort the edges in the batch (by source)
                if (!sorted)
                {
                    WharfMH::sort_edge_batch_by_source(edges, m, nn);
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

                    new_verts[i] = make_pair(v, VertexEntry(types::CompressedEdges(S, v, fl), new SamplerManager(0)));
                });

                types::MapOfChanges rewalk_points = types::MapOfChanges();

                auto replace = [&,run_seq] (const intV& v, const VertexEntry& a, const VertexEntry& b)
                {
                    auto difference_edge_tree = tree_plus::difference(b.compressed_edges, a.compressed_edges, v, run_seq);

                    lists::deallocate(a.compressed_edges.plus);
                    tree_plus::Tree_GC::decrement_recursive(a.compressed_edges.root, run_seq);

                    lists::deallocate(b.compressed_edges.plus);
                    tree_plus::Tree_GC::decrement_recursive(b.compressed_edges.root, run_seq);

					// _________________________________________________
					walk_update_time_on_delete.start();
					// use the inverted index somehow here
					if (this->walk_index.contains(v))
					{
						auto set_wids = this->walk_index.find(v);
//						cout << "set of " << v << " has " << set_wids.size() << " elements." << endl;

						// Go and visit only the wids in the inverted index to construct the MAV
						for (auto& wid : set_wids)
						{
							auto cur_walk = this->walk_storage.find(wid);
							for (types::Position position = 0; position < cur_walk.size(); position++)
							{
								if (cur_walk[position] != v) continue;

								if (!rewalk_points.contains(wid))
									rewalk_points.insert(wid, position);
								else
								{
									types::Position current_min_pos = rewalk_points.find(wid);

	                                if (current_min_pos > position)
	                                    rewalk_points.update(wid, position);
								}
							}
						}
					}

//                    for(auto& element : this->walk_storage.lock_table())
//                    {
//                        for (types::Position position = 0; position < element.second.size(); position++)
//                        {
//                            if (element.second[position] != v) continue;
//
//                            if (!rewalk_points.contains(element.first))
//                            {
//                                rewalk_points.insert(element.first, position);
//                            }
//                            else
//                            {
//                                types::Position current_min_pos = rewalk_points.find(element.first);
//
//                                if (current_min_pos > position)
//                                {
//                                    rewalk_points.update(element.first, position);
//                                }
//                            }
//                        }
//                    }
					walk_update_time_on_delete.stop();
					// __________________________________________________________

                    return VertexEntry(difference_edge_tree, a.sampler_manager);
                };

                #ifdef WHARFMH_TIMER
                    graph_update_time.start();
                #endif

                this->graph_tree = Graph::Tree::multi_insert_sorted_with_values(this->graph_tree.root, new_verts, num_starts, replace, true, run_seq);

                #ifdef WHARFMH_TIMER
                    graph_update_time.stop();
                #endif

                #ifdef WHARFMH_TIMER
                    walk_update_time.start();
                #endif

				walk_update_time_on_delete.start();
                if (apply_walk_updates) this->update_walks(rewalk_points);
				walk_update_time_on_delete.stop();

                #ifdef WHARFMH_TIMER
                    walk_update_time.stop();
                #endif

                // 6. Deallocate memory
                if (num_starts > stack_size) pbbs::free_array(new_verts);
                if (edges_deduped) pbbs::free_array(edges_deduped);

                #ifdef WHARFMH_DEBUG
                    std::cout << "Rewalk points (MapOfChanges): " << std::endl;

                    auto table = rewalk_points.lock_table();

                    for(auto& item : table)
                    {
                        std::cout << "Walk ID: " << item.first
                                  << " Position: "
                                  << (int) item.second
                                  << std::endl;
                    }
                #endif

                #ifdef WHARFMH_TIMER
                    graph_update_time.reportTotal("time(seconds)");
                    walk_update_time.reportTotal("time(seconds)");
                #endif

                return rewalk_points.size();
            }

            void update_walks(types::MapOfChanges& rewalk_points)
            {
                auto affected_walks = pbbs::sequence<types::WalkID>(rewalk_points.size());
                uintV index = 0;

                for(auto& entry : rewalk_points.lock_table())
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
                        std::cerr << "Unrecognized random walking model" << std::endl;
                        std::exit(1);
                }

                parallel_for(0, affected_walks.size(), [&](auto index)
                {
                    auto current_position = rewalk_points.template find(affected_walks[index]);
//                    auto state = model->initial_state(this->walk_storage.template find(affected_walks[index])[current_position]);

					// Delete all entries of the walk from the walk index
					for (types::Position position = 0; position < config::walk_length; position++)
					{
						auto cur_vertex = this->walk_storage.find(affected_walks[index])[position];
						this->walk_index.update_fn(cur_vertex, [&](auto &set) {
                             set.erase(affected_walks[index]);
//							 cout << "erased wid-" << affected_walks[index] << " from entry nid-" << cur_vertex << endl;
						});
					}

                    auto random = config::random; // By default random initialization
                    if (config::determinism)
//                        random = utility::Random(walk_id / total_vertices);
						random = utility::Random(affected_walks[index] / number_of_vertices());
//                    types::State state  = model->initial_state(walk_id % total_vertices);
//                    types::State state  = model->initial_state(affected_walks[index] % number_of_vertices());
//					         auto state = model->initial_state(current_vertex_new_walk);
                    auto state = model->initial_state(this->walk_storage.template find(affected_walks[index])[current_position]);

					// Insert all entries of the walk into the walk index
					for (types::Position position = 0; position < config::walk_length; position++)
                    {
						if (position < current_position)
						{
							auto cur_vertex = this->walk_storage.find(affected_walks[index])[position];
							this->walk_index.update_fn(cur_vertex, [&](auto &set) {
							  set.insert(affected_walks[index]);
//							  cout << "inserted wid-" << affected_walks[index] << " from entry nid-" << cur_vertex << endl;
							});
						}
						else // position >= current_position
						{

							this->walk_storage.update_fn(affected_walks[index], [&](auto& vector)
							{
							  vector[position] = state.first;
							});

							this->walk_index.update_fn(state.first, [&](auto& set)
							{
							  set.insert(affected_walks[index]);
//							  cout << "inserted wid-" << affected_walks[index] << " from entry nid-" << state.first << endl;
							});

							// check
//							if (graph[state.first].degrees == 0)
//							{
//								cout << "ZERO DEGREES!" << endl;
//								break;
//							}

							if (!graph[state.first].samplers->contains(state.second))
							{
								graph[state.first].samplers->insert(state.second, MetropolisHastingsSampler(state, model));
							}

							// Deterministic walks or not?
							if (config::determinism)
								state = model->new_state(state, graph[state.first].neighbors[random.irand(graph[state.first].degrees)]);
//								state = model->new_state(state, graph[state.first].neighbors[0]); // todo: do not use this one
							else
								state = graph[state.first].samplers->find(state.second).sample(state, model);
						}
                    }



//					for (types::Position position = 0/*current_position*/; position < config::walk_length; position++)
//                    {
//						if (position < current_position)
//						{
//							auto cur_vertex = this->walk_storage.find(affected_walks[index])[position];
//							if (affected_walks[index] == 1)
//								cout << "(PRE)(wid-1)current vertex= " << cur_vertex << endl;
//							prefix_set.insert(cur_vertex);
////							cout << "wid=" << index << " has vertex " << cur_vertex << " at position " << (int)position << endl;
//						}
//						else // position >= current_position
//						{
//							// -------
//							// update the walk index accordingly
//							// -------
////							auto cur_vertex = this->walk_storage.find(affected_walks[index])[position];
////							if (affected_walks[index] == 1)
////								cout << "(UPD)(wid-1)current vertex= " << cur_vertex << endl;
//////							if (position == 0)                 // ************
//////								prefix_set.insert(cur_vertex); // special case
////							if (prefix_set.find(cur_vertex) == prefix_set.end()) // vertex does not appear in the prefix of the walk. we can erase it from inv. index
////							{
////								this->walk_index.update_fn(cur_vertex, [&](auto& set)
////								{
////									set.erase(affected_walks[index]);
////									assert(set.find(affected_walks[index]) == set.end());
//////									cout << "assertion passed" << endl;
////								});
////							}
//							// ------
//
//							this->walk_storage.update_fn(affected_walks[index], [&](auto& vector)
//							{
//							  vector[position] = state.first;
//							});
//
//							if (affected_walks[index] == 1)
//								cout << "(UPD-NEW)(wid-1)current vertex= " << state.first << endl;
//
//							// ----
//							// update the walk index accordingly after the re-sampling
////							cur_vertex = this->walk_storage.find(affected_walks[index])[position]; // take again the new vertex value
//							// cur_vertex is state.first!
//							this->walk_index.update_fn(state.first, [&](auto& set)
//							{
//							  set.insert(affected_walks[index]);
//							});
//							// ---
//
//							if (!graph[state.first].samplers->contains(state.second))
//							{
//								graph[state.first].samplers->insert(state.second, MetropolisHastingsSampler(state, model));
//							}
//
//							state = graph[state.first].samplers->find(state.second).sample(state, model);
//						}
//                    }


//                    for (types::Position position = current_position; position < config::walk_length; position++)
//                    {
//                        this->walk_storage.update_fn(affected_walks[index], [&](auto& vector)
//                        {
//                            vector[position] = state.first;
//                        });
//
//                        if (!graph[state.first].samplers->contains(state.second))
//                        {
//                            graph[state.first].samplers->insert(state.second, MetropolisHastingsSampler(state, model));
//                        }
//
//                        state = graph[state.first].samplers->find(state.second).sample(state, model);
//                    }
                });
            }

		/**
         * @brief Prints memory footprint details.
         */
        void memory_footprint() //const todo: this is not needed. i do not change the cuckoo map
        {
            std::cout << std::endl;

            size_t graph_vertices = this->number_of_vertices();
            size_t graph_edges    = this->number_of_edges();

            size_t vertex_node_size = Graph::node_size();
//            size_t index_node_size  = InvertedIndex::node_size();
            size_t c_tree_node_size = types::CompressedTreesLists::node_size();

            size_t edges_heads    = 0;
            size_t edges_bytes    = 0;
            size_t samplers_bytes = 0;
            size_t flat_graph_bytes = 0;
            auto flat_graph = this->flatten_vertex_tree();

            for (auto i = 0; i < flat_graph.size(); i++)
            {
                flat_graph_bytes += sizeof(flat_graph[i]);

                edges_heads += flat_graph[i].compressed_edges.edge_tree_nodes();
                edges_bytes += flat_graph[i].compressed_edges.size_in_bytes(i);

                for (auto& entry : flat_graph[i].sampler_manager->lock_table())
                {
                    samplers_bytes += sizeof(entry.first) + sizeof(entry.second);
                }
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

//            std::cout << "Walks Trees: \n\t"
//                      << "Heads: " << InvertedIndex::used_node()
//                      << ", Head size: " << index_node_size
//                      << ", Memory usage: " << utility::MB(InvertedIndex::get_used_bytes()) << " MB"
//                      << " = " << utility::GB(InvertedIndex::get_used_bytes()) << " GB" << std::endl;

			size_t walk_seq = 0;
			for (const auto &it : walk_storage.lock_table())
			{
				walk_seq += sizeof(it.first); //sizeof(uint32_t); // sizeof(it.first);
//				cout << "wid-" << it.first << " | ";
				for (auto i = it.second.begin(); i != it.second.end(); i++)
				{
					walk_seq += sizeof(*i); //sizeof(uint32_t); // sizeof(*i);
//					cout << *i << " ";
				}
//				cout << endl;
			}
//			assert(new_walk_storage.size() == walk_storage.size());

            std::cout << "Walks (sequences): \n\t"
                      << "Memory usage: " << utility::MB(walk_seq) << " MB"
                      << " = " << utility::GB(walk_seq) << " GB" << std::endl;

			size_t walk_ind = 0;
			for (const auto &it : walk_index.lock_table())
			{
				walk_ind += sizeof(it.first); //sizeof(uint32_t); // sizeof(it.first);
				for (auto i = it.second.begin(); i != it.second.end(); i++)
				{
					walk_ind += sizeof(*i); // sizeof(uint32_t); // sizeof(*i);
				}
			}

			std::cout << "Walks Inverted Index: \n\t"
			          << "Memory usage: " << utility::MB(walk_ind) << " MB"
			          << " = " << utility::GB(walk_ind) << " GB" << std::endl;

            std::cout << "Samplers: \n\t"
                      << "Total memory usage: " << utility::MB(samplers_bytes)
                      << " MB = " << utility::GB(samplers_bytes)
                      << " GB" << std::endl;

            std::cout << "Flat graph: \n\t"
                      << "Total memory usage: " << utility::MB(flat_graph_bytes)
                      << " MB = " << utility::GB(flat_graph_bytes)
                      << " GB" << std::endl;

            size_t total_memory = Graph::get_used_bytes() + (walk_seq + walk_ind) //InvertedIndex::get_used_bytes()
                    + edges_bytes + edges_heads*c_tree_node_size + samplers_bytes;

            std::cout << "Total memory used: \n\t" << utility::MB(total_memory) << " MB = "
                      << utility::GB(total_memory) << " GB" << std::endl;

            std::cout << std::endl;
        }

        private:
            Graph graph_tree;
            WalkStorage walk_storage;
			WalkIndex   walk_index;  // simplistic inverted index structure {nid, set of wids}

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
                #ifdef WHARFMH_TIMER
                    timer timer("WharfMH::SortEdgeBatchBySource");
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

                #ifdef WHARFMH_TIMER
                    timer.reportTotal("time (seconds)");
                #endif
            }
    };
}

#endif // DYNAMIC_GRAPH_REPRESENTATION_LEARNING_WITH_METROPOLIS_HASTINGS_WharfMH_H
