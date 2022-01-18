#include <gtest/gtest.h>

#include <wharf.h>

class WharfTest : public testing::Test
{
    public:
        void SetUp()    override;
        void TearDown() override;

    protected:
        long total_vertices;
        long total_edges;
        uintV* edges;
        uintE* offsets;
        bool mmap = false;
        bool is_symmetric = true;
//        std::string default_file_path = "data/email-graph";
//        std::string default_file_path = "data/flickr-graph";
        std::string default_file_path = "data/aspen-paper-graph";
};

void WharfTest::SetUp()
{
    std::cout << "-----------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "Malin running with " << num_workers() << " threads" << std::endl;

    // transform an input graph file into an adjacency graph format
    std::string command = "./SNAPtoAdj -s -f " + this->default_file_path + " data/adjacency-graph-format.txt";
    int result = system(command.c_str());

    if (result != 0)
    {
        std::cerr << "MalinTest::SetUp::Input file could not be transformed!" << std::endl;
        exit(1);
    }

    std::tie(total_vertices, total_edges, offsets, edges) = read_unweighted_graph("data/adjacency-graph-format.txt", is_symmetric, mmap);
    std::cout << std::endl;
}

void WharfTest::TearDown()
{
    // remove adjaceny graph format representation
    int graph = system("rm -rf data/adjacency-graph-format.txt");

    if (graph != 0)
    {
        std::cerr << "MalinTest::TearDown::Could not remove static graph input file" << std::endl;
    }

    std::cout << "-----------------------------------------------------------------------------------------------------" << std::endl;
}

TEST_F(WharfTest, MalinConstructor)
{
    dygrl::Wharf malin = dygrl::Wharf(total_vertices, total_edges, offsets, edges, false);

    // assert the number of vertices and edges in a graph
    ASSERT_EQ(malin.number_of_vertices(), total_vertices);
    ASSERT_EQ(malin.number_of_edges(), total_edges);

    // construct a flat snapshot of a graph
    auto flat_snapshot = malin.flatten_vertex_tree();

    // assert
    parallel_for(0, total_vertices, [&] (long i)
    {
        size_t off = offsets[i];
        size_t degree = ((i == (total_vertices - 1)) ? total_edges : offsets[i+1]) - off;
        auto S = pbbs::delayed_seq<uintV>(degree, [&] (size_t j) { return edges[off + j]; });

        // assert expected degrees
        ASSERT_EQ(flat_snapshot[i].compressed_edges.degree(), degree);

        // assert that compressed_walks_tree is empty
//        ASSERT_EQ(flat_snapshot[i].compressed_walks.root, nullptr); // todo: removed it
        ASSERT_EQ(flat_snapshot[i].compressed_walks.size(), 0);

        // assert empty samplers
        ASSERT_EQ(flat_snapshot[i].sampler_manager->size(), 0);

        // assert expected neighbours
        auto edges = flat_snapshot[i].compressed_edges.get_edges(i);

        for(auto j = 0; j < degree; j++)
        {
            bool flag = false;

            for (auto k = 0; k <  S.size(); k++)
            {
                if (S[k] == edges[j]) flag = true;
            }

            ASSERT_EQ(flag, true);
        }
    });
}

TEST_F(WharfTest, DockDestructor)
{
    dygrl::Wharf malin = dygrl::Wharf(total_vertices, total_edges, offsets, edges);

    malin.print_memory_pool_stats();
    malin.destroy();
    malin.print_memory_pool_stats();

    // assert vertices and edges
    ASSERT_EQ(malin.number_of_vertices(), 0);
    ASSERT_EQ(malin.number_of_edges(), 0);

    // construct a flat snapshot of a graph
    auto flat_snapshot = malin.flatten_vertex_tree();

    // assert that flat snapshot does not exits
    ASSERT_EQ(flat_snapshot.size(), 0);
}

TEST_F(WharfTest, MalinDestroyIndex)
{
    dygrl::Wharf malin = dygrl::Wharf(total_vertices, total_edges, offsets, edges);
    malin.generate_initial_random_walks();

    malin.print_memory_pool_stats();
    malin.destroy_index();
    malin.print_memory_pool_stats();

    // assert vertices and edges
    ASSERT_EQ(malin.number_of_vertices(), total_vertices);
    ASSERT_EQ(malin.number_of_edges(), total_edges);

    // construct a flat snapshot of a graph
    auto flat_snapshot = malin.flatten_vertex_tree();

    parallel_for(0, total_vertices, [&] (long i)
    {
//        ASSERT_EQ(flat_snapshot[i].compressed_walks.size(), 0); // todo: removed it
        ASSERT_EQ(flat_snapshot[i].sampler_manager->size(), 0);
    });
}

TEST_F(WharfTest, InsertBatchOfEdges)
{
    // create wharf instance (vertices & edges)
    dygrl::Wharf malin = dygrl::Wharf(total_vertices, total_edges, offsets, edges);
    auto start_edges = malin.number_of_edges();

    // geneate edges
    auto edges = utility::generate_batch_of_edges(1000000, malin.number_of_vertices(), false, false);

    // insert batch of edges
    malin.insert_edges_batch(edges.second, edges.first, true, false, std::numeric_limits<size_t>::max(), false);

    std::cout << "Edges before batch insert: " << start_edges << std::endl;
    std::cout << "Edges after batch insert: "  << malin.number_of_edges() << std::endl;

    // assert edge insertion
    ASSERT_GE(malin.number_of_edges(), start_edges);
}

// Pump up the test to debug diff updates approach
TEST_F(WharfTest, InsertBatchOfEdgesPlayground)
{
	// create wharf instance (vertices & edges)
	dygrl::Wharf malin = dygrl::Wharf(total_vertices, total_edges, offsets, edges);
	auto start_edges = malin.number_of_edges();

	malin.generate_initial_random_walks();

	// geneate edges
	auto edges = utility::generate_batch_of_edges(10, malin.number_of_vertices(), 10, false, false);

	// insert batch of edges
	malin.insert_edges_batch(edges.second, edges.first, true, false/*, std::numeric_limits<size_t>::max(), false*/);

	auto flat_graph = malin.flatten_vertex_tree();
	for (auto i = 0; i < malin.number_of_vertices(); i++)
	{
		cout << "vertex " << i << endl;
//		flat_graph[i].compressed_edges.iter_elms(i, [&](auto edge){
//			cout << edge << " ";
//		});
//		cout << endl;

		int inc = 0;
		for (auto wt = flat_graph[i].compressed_walks.begin(); wt != flat_graph[i].compressed_walks.end(); wt++)
		{
			inc++;
			cout << "walk-tree " << inc << endl;
			wt->iter_elms(i, [&](auto enc_triplet){
				auto pair = pairings::Szudzik<types::Vertex>::unpair(enc_triplet);

				auto walk_id  = pair.first / config::walk_length;                  // todo: needs floor?
				auto position = pair.first - (walk_id * config::walk_length); // todo: position here starts from 0. verify this one!
				auto next_vertex   = pair.second;
//				cout << enc_triplet << " ";
				cout << "{" << walk_id << ", " << position << ", " << next_vertex << "}" << " " << endl;
			});
			cout << endl;
		}
	}

//	// geneate edges
//	auto edges = utility::generate_batch_of_edges(1000000, malin.number_of_vertices(), false, false);
//
//	// insert batch of edges
//	malin.insert_edges_batch(edges.second, edges.first, true, false, std::numeric_limits<size_t>::max(), false);

	std::cout << "Edges before batch insert: " << start_edges << std::endl;
	std::cout << "Edges after batch insert: "  << malin.number_of_edges() << std::endl;

	// assert edge insertion
	ASSERT_GE(malin.number_of_edges(), start_edges);
}

TEST_F(WharfTest, DeleteBatchOfEdges)
{
    // create wharf instance (vertices & edges)
    dygrl::Wharf malin = dygrl::Wharf(total_vertices, total_edges, offsets, edges);
    auto start_edges = malin.number_of_edges();

    // geneate edges
    auto edges = utility::generate_batch_of_edges(1000000, malin.number_of_vertices(), false, false);

    // insert batch of edges
    malin.delete_edges_batch(edges.second, edges.first, true, false, std::numeric_limits<size_t>::max(), false);

    std::cout << "Edges before batch delete: " << start_edges << std::endl;
    std::cout << "Edges after batch delete: "  << malin.number_of_edges() << std::endl;

    // assert edge deletion
    ASSERT_LE(malin.number_of_edges(), start_edges);
}

TEST_F(WharfTest, UpdateRandomWalksOnInsertEdges)
{
    // create graph and walks
    dygrl::Wharf malin = dygrl::Wharf(total_vertices, total_edges, offsets, edges);
    malin.generate_initial_random_walks();

    // print random walks
//    for(int i = 0; i < config::walks_per_vertex * malin.number_of_vertices(); i++)
//        std::cout << malin.walk(i) << std::endl;

    // geneate edges
    auto edges = utility::generate_batch_of_edges(1000, malin.number_of_vertices(), false, false);

    // insert batch of edges
    malin.insert_edges_batch(edges.second, edges.first, true, false);

    // print updated random walks
//    for(int i = 0; i < config::walks_per_vertex * malin.number_of_vertices(); i++)
//        std::cout << malin.walk(i) << std::endl;
}

TEST_F(WharfTest, UpdateRandomWalksOnDeleteEdges)
{
    // create graph and walks
    dygrl::Wharf malin = dygrl::Wharf(total_vertices, total_edges, offsets, edges);
    malin.generate_initial_random_walks();

    // print random walks
//    for(int i = 0; i < config::walks_per_vertex * malin.number_of_vertices(); i++)
//        std::cout << malin.walk(i) << std::endl;

    // geneate edges
    auto edges = utility::generate_batch_of_edges(1000, malin.number_of_vertices(), false, false);

    // insert batch of edges
    malin.delete_edges_batch(edges.second, edges.first, true, false);

    // print updated random walks
//    for(int i = 0; i < config::walks_per_vertex * malin.number_of_vertices(); i++)
//        std::cout << malin.walk(i) << std::endl;
}

TEST_F(WharfTest, UpdateRandomWalks)
{
    // create graph and walks
    dygrl::Wharf malin = dygrl::Wharf(total_vertices, total_edges, offsets, edges);
    malin.generate_initial_random_walks();

    // print random walks
//    for(int i = 0; i < config::walks_per_vertex * malin.number_of_vertices(); i++)
//        std::cout << malin.walk(i) << std::endl;

    for(int i = 0; i < 10; i++)
    {
        // geneate edges //todo: what happens if I generate a high number of edges?
        auto edges = utility::generate_batch_of_edges(1000, malin.number_of_vertices(), false, false);

        malin.insert_edges_batch(edges.second, edges.first, true, false);
        malin.delete_edges_batch(edges.second, edges.first, true, false);
    }

    // print random walks
//    for(int i = 0; i < config::walks_per_vertex * malin.number_of_vertices(); i++)
//        std::cout << malin.walk(i) << std::endl;
}

// -------------------------------//
// ---- tests for range search ---//
// -------------------------------//

TEST_F(WharfTest, GenerateAndPrintInitialRW)
{
    // create graph and walks
    dygrl::Wharf malin = dygrl::Wharf(total_vertices, total_edges, offsets, edges);
    malin.generate_initial_random_walks();

    auto walk_printing_timer = timer("TotalTimeToRewalk", false);
    double time_simple, time_range;

    walk_printing_timer.start();
    // for debugging purposes - uses simple find_next only
//    for(int i = 0; i < config::walks_per_vertex * malin.number_of_vertices(); i++)
//        std::cout << "simple - id=" << i << ":\t" << malin.walk_simple_find(i) << std::endl;
    time_simple = walk_printing_timer.get_total();

    cout << "-- now print out the walks with the find_in_range operation" << endl;

    walk_printing_timer.reset(); walk_printing_timer.start();
    // print random walks
//    for(int i = 0; i < config::walks_per_vertex * malin.number_of_vertices(); i++)
//        std::cout << "range - id=" << i << ":\t" << malin.walk(i) << std::endl;
    time_range = walk_printing_timer.get_total();

    cout << "Time to print all walk corpus with simple find: " << time_simple << endl
         << " and to print all walk corpus with range  find: " << time_range  << endl;
}

// -----------------------------------
// -----------------------------------

TEST_F(WharfTest, UpdateRandomWalksWithRangeSearch)
{
    // create graph and walks
    dygrl::Wharf malin = dygrl::Wharf(total_vertices, total_edges, offsets, edges);
    malin.generate_initial_random_walks();

    // print random walks before batch insertion
//    for(int i = 0; i < config::walks_per_vertex * malin.number_of_vertices(); i++)
//        std::cout << "id=" << i << ":\t" << malin.walk(i) << std::endl;

    // geneate edges
    auto edges = utility::generate_batch_of_edges(1000, malin.number_of_vertices(), false, false);
//    for (auto i = 0; i < edges.second; i++)
//        cout << "edge-" << i + 1 << " is [" << get<0>(edges.first[i]) << ", " << get<1>(edges.first[i]) << "]" << endl;
    // insert batch of edges
    malin.insert_edges_batch(edges.second, edges.first, true, false);

    // print random walks after batch insertion
//    for(int i = 0; i < config::walks_per_vertex * malin.number_of_vertices(); i++)
//        std::cout << "id=" << i << ":\t" << malin.walk(i) << std::endl;

//    cout << "and now with simple find" << endl;
//    // print random walks after batch insertion
//    for(int i = 0; i < config::walks_per_vertex * malin.number_of_vertices(); i++)
//        std::cout << "id=" << i << ":\t" << malin.walk_simple_find(i) << std::endl;
}


TEST_F(WharfTest, MalinThroughputLatency)
{
	dygrl::Wharf malin = dygrl::Wharf(total_vertices, total_edges, offsets, edges);
	malin.generate_initial_random_walks();
	int n_trials = 3; //3;

//	cout << "total vertices: " << total_vertices << endl;
//	cout << "total edges:    " << total_edges << endl;

//	double limit = 5.5;

//		WharfMH.walk_cout(1);
//		cout << WharfMH.walk(1);
//		WharfMH.walk_cout(13);

// ----------------------------------------------
//	cout << "WALKS" << endl;
//	for (auto i = 0; i < total_vertices * config::walks_per_vertex; i++)
//		cout << malin.walk(i) << endl;
// ----------------------------------------------


//		exit(1);

	// ----------------------------------------
	// Store the (min, max) bounds of each walk-tree in the initial walk corpus to reassign them after each run
	using minMaxPair = std::pair<types::Vertex, types::Vertex>;
	auto initial_minmax_bounds = pbbs::sequence<minMaxPair>(total_vertices);

	// construct a flat snapshot of a graph
	auto flat_snapshot = malin.flatten_vertex_tree();

	// Cache the initial ranges
//	parallel_for(0, total_vertices, [&] (auto i) {
//	  auto min = flat_snapshot[i].compressed_walks.vnext_min;
//	  auto max = flat_snapshot[i].compressed_walks.vnext_max;
//	  initial_minmax_bounds[i] = std::make_pair(min, max);
////        cout << "vertex=" << i << " {min=" << min << ", max=" << max << "}" << endl;
//	});
	// -------------------------------------


	auto batch_sizes = pbbs::sequence<size_t>(1);
	batch_sizes[0] = 500; //5;
//	batch_sizes[1] = 50;
//	batch_sizes[2] = 500;
//	batch_sizes[3] = 5000;
//	batch_sizes[4] = 50000;
//    batch_sizes[5] = 500000;

	for (short int i = 0; i < batch_sizes.size(); i++)
	{
		timer insert_timer("InsertTimer");
		timer delete_timer("DeleteTimer");

		graph_update_time_on_insert.reset();
		walk_update_time_on_insert.reset();
		graph_update_time_on_delete.reset();
		walk_update_time_on_delete.reset();
		// --- profiling initialization
		walk_insert_init.reset();
		walk_insert_2jobs.reset();
		walk_insert_2accs.reset();
		ij.reset();
		dj.reset();
		walk_find_in_vertex_tree.reset();
		walk_find_next_tree.reset();
		szudzik_hash.reset();
		fnir_tree_search.reset();
		// ---

		std::cout << "Batch size = " << 2 * batch_sizes[i] << " | ";

		double last_insert_time = 0;
		double last_delete_time = 0;

		auto latency_insert = pbbs::sequence<double>(n_trials);
		auto latency_delete = pbbs::sequence<double>(n_trials);
		auto latency        = pbbs::sequence<double>(n_trials);

		double total_insert_walks_affected = 0;
		double total_delete_walks_affected = 0;

		int batch_seed[n_trials];
		for (auto i = 0; i < n_trials; i++)
			batch_seed[i] = i; // say the seed equals to the #trial

		for (short int trial = 0; trial < n_trials; trial++)
		{
			cout << "trial-" << trial << " and batch_seed-" << batch_seed[trial] << endl;


			// Check whether the bound for min and max are correctly resetted
//			parallel_for(0, total_vertices, [&] (auto i) {
//			  assert(flat_snapshot[i].compressed_walks.vnext_min == get<0>(initial_minmax_bounds[i]));
//			  assert(flat_snapshot[i].compressed_walks.vnext_max == get<1>(initial_minmax_bounds[i]));
//			});

			size_t graph_size_pow2 = 1 << (pbbs::log2_up(total_vertices) - 1);
			auto edges = utility::generate_batch_of_edges(batch_sizes[i], total_vertices, batch_seed[trial], false, false);

			std::cout << edges.second << " ";
			// ---
//		    for (auto i = 0; i < edges.second; i++)
//		        cout << "edge-" << i + 1 << " is [" << get<0>(edges.first[i]) << ", " << get<1>(edges.first[i]) << "]" << endl;

			insert_timer.start();
			auto x = malin.insert_edges_batch(edges.second, edges.first, false, true, graph_size_pow2);
			insert_timer.stop();

			total_insert_walks_affected += x.size();

			last_insert_time = walk_update_time_on_insert.get_total() - last_insert_time;
			latency_insert[trial] = (double) last_insert_time / x.size();

            delete_timer.start();
            auto y = malin.delete_edges_batch(edges.second, edges.first, false, true, graph_size_pow2);
            delete_timer.stop();

            total_delete_walks_affected += y.size();

            last_delete_time = walk_update_time_on_delete.get_total() - last_delete_time;
            latency_delete[trial] = (double) last_delete_time / y.size();

            latency[trial] = (double) (last_insert_time + last_delete_time) / (x.size() + y.size());
//			latency[trial] = latency_insert[trial]; // todo: for now latency insert trial only

//            if (insert_timer.get_total() > 2*limit || delete_timer.get_total() > 2*limit) goto endloop;

			// free edges
			pbbs::free_array(edges.first);

			// Reset the initial corpus next vertex bounds
//			parallel_for(0, total_vertices, [&] (auto i) {
//			  flat_snapshot[i].compressed_walks.vnext_min = get<0>(initial_minmax_bounds[i]);
//			  flat_snapshot[i].compressed_walks.vnext_max = get<1>(initial_minmax_bounds[i]);
//			});
		}

		std::cout << std::endl;

		std::cout << "Average insert time = " << insert_timer.get_total() / n_trials << std::endl;
		std::cout << "Average graph update insert time = " << graph_update_time_on_insert.get_total() / n_trials << std::endl;
		std::cout << "Average walk update insert time = " << walk_update_time_on_insert.get_total() / n_trials
		          << ", average walk affected = " << total_insert_walks_affected / n_trials << std::endl;

        std::cout << "Average delete time = " << delete_timer.get_total() / n_trials << std::endl;
        std::cout << "Average graph update delete time = " << graph_update_time_on_delete.get_total() / n_trials << std::endl;
        std::cout << "Average walk update delete time = " << walk_update_time_on_delete.get_total() / n_trials
                  << ", average walk affected = " << total_delete_walks_affected / n_trials << std::endl;

		// --- profiling ---
		std::cout << "{ total profiling for insert and delete" << std::endl;
		std::cout << "Initialization: " << walk_insert_init.get_total() / n_trials << " (" << (walk_insert_init.get_total()*100) / (walk_insert_init.get_total() +
		                                                                                                                            walk_insert_2jobs.get_total() +
		                                                                                                                            walk_insert_2accs.get_total()) << "%)" << std::endl;
		std::cout << "Insert/Delete Jobs: " << walk_insert_2jobs.get_total() / n_trials << " (" << (walk_insert_2jobs.get_total()*100) / (walk_insert_init.get_total() +
		                                                                                                                                  walk_insert_2jobs.get_total() +
		                                                                                                                                  walk_insert_2accs.get_total()) << "%)" << std::endl;
		std::cout << "InsertJob: " << ij.get_total() / n_trials << " | DeleteJob: " << dj.get_total() / n_trials << std::endl;
		std::cout << "FindInVertexTree in DeleteJob total: " << walk_find_in_vertex_tree.get_total() / n_trials << std::endl;
		std::cout << "FindNext in DeleteJob total: " << walk_find_next_tree.get_total() / n_trials << std::endl;
		std::cout << "FindNext (search of the tree): " << fnir_tree_search.get_total() / n_trials << std::endl;
		std::cout << "Sudzik total: " << szudzik_hash.get_total() / n_trials << std::endl;

		std::cout << "Accumulators: " << walk_insert_2accs.get_total() / n_trials << " (" << (walk_insert_2accs.get_total()*100) / (walk_insert_init.get_total() +
		                                                                                                                            walk_insert_2jobs.get_total() +
		                                                                                                                            walk_insert_2accs.get_total()) << "%)" << std::endl;
		std::cout << "}" << std::endl;
		// --- profiling ---

		// latencies
		std::cout << "Average walk insert latency = { ";
		for(int i = 0; i < n_trials; i++)
		{
			std::cout << latency_insert[i] << " ";
		}
		std::cout << "}" << std::endl;

        std::cout << "Average walk delete latency = { ";
        for(int i = 0; i < n_trials; i++)
        {
            std::cout << latency_delete[i] << " ";
        }
        std::cout << "}" << std::endl;

		std::cout << "Average walk update latency = { ";
		for(int i = 0; i < n_trials; i++)
		{
			std::cout << latency[i] << " ";
		}
		std::cout << "}" << std::endl;
	}

	auto flat_graph = malin.flatten_vertex_tree();
	for (auto i = 0; i < malin.number_of_vertices(); i++)
	{
		cout << "vertex " << i << endl;
//		flat_graph[i].compressed_edges.iter_elms(i, [&](auto edge){
//			cout << edge << " ";
//		});
//		cout << endl;

		cout << "size of walk-tree vector " << flat_graph[i].compressed_walks.size() << endl;
		int inc = 0;
		for (auto wt = flat_graph[i].compressed_walks.begin(); wt != flat_graph[i].compressed_walks.end(); wt++) // print the walk-trees in chronological order
		{
			inc++;
			cout << "walk-tree " << inc << endl;
			wt->iter_elms(i, [&](auto enc_triplet){
			  auto pair = pairings::Szudzik<types::Vertex>::unpair(enc_triplet);

			  auto walk_id  = pair.first / config::walk_length;                  // todo: needs floor?
			  auto position = pair.first - (walk_id * config::walk_length); // todo: position here starts from 0. verify this one!
			  auto next_vertex   = pair.second;
//				cout << enc_triplet << " ";
			  cout << "{" << walk_id << ", " << position << ", " << next_vertex << "}" << " " << endl;
			});
			cout << endl;
		}
	}

// ----------------------------------------------
//	cout << "(NEW) WALKS" << endl;
//	for (auto i = 0; i < total_vertices * config::walks_per_vertex; i++)
//		cout << malin.walk(i) << endl;
// ----------------------------------------------

	endloop:
	std::cout << "Loop ended" << std::endl;
}

TEST_F(WharfTest, BatchGenerator)
{
	cout << "first batch" << endl;
	auto edges = utility::generate_batch_of_edges(10, 6, false, false);
	for (auto i = 0; i < edges.second; i++)
		cout << "edge-" << i + 1 << " is [" << get<0>(edges.first[i]) << ", " << get<1>(edges.first[i]) << "]" << endl;

	cout << "second batch" << endl;
	edges = utility::generate_batch_of_edges(10, 6, false, false);
	for (auto i = 0; i < edges.second; i++)
		cout << "edge-" << i + 1 << " is [" << get<0>(edges.first[i]) << ", " << get<1>(edges.first[i]) << "]" << endl;

	cout << "third batch" << endl;
	edges = utility::generate_batch_of_edges(10, 6, false, false);
	for (auto i = 0; i < edges.second; i++)
		cout << "edge-" << i + 1 << " is [" << get<0>(edges.first[i]) << ", " << get<1>(edges.first[i]) << "]" << endl;
}

// --------------------------------------------------------------------------
// --------- WHARF++ tests --------------------------------------------------
// --------------------------------------------------------------------------

// Pump up the test to debug diff updates approach
TEST_F(WharfTest, WharfPlusPlusPlayground)
{
	// create wharf instance (vertices & edges)
	dygrl::Wharf malin = dygrl::Wharf(total_vertices, total_edges, offsets, edges);
	auto start_edges = malin.number_of_edges();

	malin.generate_initial_random_walks();

	// geneate edges
	auto edges = utility::generate_batch_of_edges(1, malin.number_of_vertices(), 10, false, false);
	for (auto i = 0; i < edges.second; i++)
		cout << "[" << get<0>(edges.first[i]) << ", " << get<1>(edges.first[i]) << "]" << endl;

	// insert batch of edges
	malin.insert_edges_batch(edges.second, edges.first, true, false);

	auto flat_graph = malin.flatten_vertex_tree();
	for (auto i = 0; i < malin.number_of_vertices(); i++)
	{
		cout << "vertex " << i << endl;
//		flat_graph[i].compressed_edges.iter_elms(i, [&](auto edge){
//			cout << edge << " ";
//		});
//		cout << endl;

		cout << "size of walk-tree vector " << flat_graph[i].compressed_walks.size() << endl;
		int inc = 0;
		for (auto wt = flat_graph[i].compressed_walks.begin(); wt != flat_graph[i].compressed_walks.end(); wt++) // print the walk-trees in chronological order
		{
			inc++;
			cout << "walk-tree " << inc << endl;
			wt->iter_elms(i, [&](auto enc_triplet){
			  auto pair = pairings::Szudzik<types::Vertex>::unpair(enc_triplet);

			  auto walk_id  = pair.first / config::walk_length;                  // todo: needs floor?
			  auto position = pair.first - (walk_id * config::walk_length); // todo: position here starts from 0. verify this one!
			  auto next_vertex   = pair.second;
//				cout << enc_triplet << " ";
			  cout << "{" << walk_id << ", " << position << ", " << next_vertex << "}" << " " << endl;
			});
			cout << endl;
		}
	}

//	// geneate edges
//	auto edges = utility::generate_batch_of_edges(1000000, malin.number_of_vertices(), false, false);
//
//	// insert batch of edges
//	malin.insert_edges_batch(edges.second, edges.first, true, false, std::numeric_limits<size_t>::max(), false);

	std::cout << "Edges before batch insert: " << start_edges << std::endl;
	std::cout << "Edges after batch insert: "  << malin.number_of_edges() << std::endl;

	// assert edge insertion
	ASSERT_GE(malin.number_of_edges(), start_edges);
}

// ----------------------------
// --- INSERT ONLY WORKLOAD ---
// ----------------------------

TEST_F(WharfTest, WharfInsertOnlyWorkload) {
	dygrl::Wharf malin = dygrl::Wharf(total_vertices, total_edges, offsets, edges);
	malin.generate_initial_random_walks();
	int n_batches = 3; // todo: how many batches per batch size?

	// TODO: Why incorrect numbers when MALIN_DEBUG is off?

	auto batch_sizes = pbbs::sequence<size_t>(1);
	batch_sizes[0] = 5; //5;
//	batch_sizes[1] = 50;
//	batch_sizes[2] = 500;
//	batch_sizes[3] = 5000;
//	batch_sizes[4] = 50000;
//  batch_sizes[5] = 500000;

	for (short int i = 0; i < batch_sizes.size(); i++)
	{
		timer insert_timer("InsertTimer");
		timer delete_timer("DeleteTimer");

		graph_update_time_on_insert.reset();
		walk_update_time_on_insert.reset();
		graph_update_time_on_delete.reset();
		walk_update_time_on_delete.reset();
		// --- profiling initialization
		walk_insert_init.reset();
		walk_insert_2jobs.reset();
		walk_insert_2accs.reset();
		ij.reset();
		dj.reset();
		walk_find_in_vertex_tree.reset();
		walk_find_next_tree.reset();
		szudzik_hash.reset();
		fnir_tree_search.reset();
		MAV_time.reset();
		read_access_MAV.reset();
		// ---

		std::cout << "Batch size = " << 2 * batch_sizes[i] << " | ";

		double last_insert_time = 0;

		auto latency_insert = pbbs::sequence<double>(n_batches);
		auto latency = pbbs::sequence<double>(n_batches);

		double total_insert_walks_affected = 0;
		double total_delete_walks_affected = 0;

		int batch_seed[n_batches];
		for (auto i = 0; i < n_batches; i++)
			batch_seed[i] = i; // say the seed equals to the #batch todo: produce a different batch each time

		for (short int b = 0; b < n_batches; b++)
		{
			cout << "batch-" << b << " and batch_seed-" << batch_seed[b] << endl;

			size_t graph_size_pow2 = 1 << (pbbs::log2_up(total_vertices) - 1);
			auto edges = utility::generate_batch_of_edges(batch_sizes[i], total_vertices, batch_seed[b], false, false);

			std::cout << edges.second << " ";
//		    for (auto i = 0; i < edges.second; i++)
//		        cout << "edge-" << i + 1 << " is [" << get<0>(edges.first[i]) << ", " << get<1>(edges.first[i]) << "]" << endl;

cout << "10" << endl;
			insert_timer.start();
			auto x = malin.insert_edges_batch(edges.second, edges.first, b+1, false, true, graph_size_pow2); // pass the batch number as well
			insert_timer.stop();
cout << "11" << endl;
			total_insert_walks_affected += x.size();

			last_insert_time = walk_update_time_on_insert.get_total() - last_insert_time;
			latency_insert[b] = (double) last_insert_time / x.size();

			latency[b] = latency_insert[b];

			// free edges
			pbbs::free_array(edges.first);
		}
		cout << fixed;
		std::cout << std::endl;

		std::cout << "Average insert time = "
		          << insert_timer.get_total() / n_batches << std::endl;
		std::cout << "Average graph update insert time = "
		          << graph_update_time_on_insert.get_total() / n_batches
		          << std::endl;
		std::cout << "Average walk update insert time = "
		          << walk_update_time_on_insert.get_total() / n_batches
		          << ", average walk affected = "
		          << total_insert_walks_affected / n_batches << std::endl;

		std::cout << "Average delete time = "
		          << delete_timer.get_total() / n_batches << std::endl;
		std::cout << "Average graph update delete time = "
		          << graph_update_time_on_delete.get_total() / n_batches
		          << std::endl;
		std::cout << "Average walk update delete time = "
		          << walk_update_time_on_delete.get_total() / n_batches
		          << ", average walk affected = "
		          << total_delete_walks_affected / n_batches << std::endl;

		// MAV time
		std::cout << "Average MAV (we are not deleting obsolete parts) = "
		          << MAV_time.get_total() / n_batches
		          << std::endl;
		// read access time MAV
		std::cout << "Average Read Access Time MAV = "
		          << read_access_MAV.get_total() / n_batches
		          << std::endl;

		std::cout << "Total MAV (we are not deleting obsolete parts) = " << MAV_time.get_total() << std::endl;
		std::cout << "Total Read Access Time MAV = " << read_access_MAV.get_total() << std::endl;
		std::cout << "Total walk update insert time = " << walk_update_time_on_insert.get_total() << ", average walk affected = " << total_insert_walks_affected / n_batches << std::endl;
		std::cout << "Total #sampled vertices = " << malin.number_of_sampled_vertices << std::endl;

		// --- profiling ---
		std::cout << "{ total profiling for insert and delete" << std::endl;
		std::cout << "Initialization: "
		          << walk_insert_init.get_total() / n_batches << " ("
		          << (walk_insert_init.get_total() * 100) /
		             (walk_insert_init.get_total() +
		              walk_insert_2jobs.get_total() +
		              walk_insert_2accs.get_total()) << "%)" << std::endl;
		std::cout << "Insert/Delete Jobs: "
		          << walk_insert_2jobs.get_total() / n_batches << " ("
		          << (walk_insert_2jobs.get_total() * 100) /
		             (walk_insert_init.get_total() +
		              walk_insert_2jobs.get_total() +
		              walk_insert_2accs.get_total()) << "%)" << std::endl;
		std::cout << "InsertJob: " << ij.get_total() / n_batches
		          << " | DeleteJob: " << dj.get_total() / n_batches << std::endl;
		std::cout << "FindInVertexTree in DeleteJob total: "
		          << walk_find_in_vertex_tree.get_total() / n_batches
		          << std::endl;
		std::cout << "FindNext in DeleteJob total: "
		          << walk_find_next_tree.get_total() / n_batches << std::endl;
		std::cout << "FindNext (search of the tree): "
		          << fnir_tree_search.get_total() / n_batches << std::endl;
		std::cout << "Sudzik total: " << szudzik_hash.get_total() / n_batches
		          << std::endl;

		std::cout << "Accumulators: "
		          << walk_insert_2accs.get_total() / n_batches << " ("
		          << (walk_insert_2accs.get_total() * 100) /
		             (walk_insert_init.get_total() +
		              walk_insert_2jobs.get_total() +
		              walk_insert_2accs.get_total()) << "%)" << std::endl;
		std::cout << "}" << std::endl;
		// --- profiling ---

		// latencies
		std::cout << "Average walk insert latency = { ";
		for (int i = 0; i < n_batches; i++) {
			std::cout << latency_insert[i] << " ";
		}
		std::cout << "}" << std::endl;

		std::cout << "Average walk update latency = { ";
		for (int i = 0; i < n_batches; i++) {
			std::cout << latency[i] << " ";
		}
		std::cout << "}" << std::endl;
	}

	// Merge all walks after the bathes
	timer MergeAll("MergeAllTimer", false);
	MergeAll.start();
	malin.merge_walk_trees_all_vertices_parallel(n_batches); // use the parallel merging
	MergeAll.stop();
	std::cout << "Merge all the walk-trees time: " << MergeAll.get_total() << std::endl;


	auto flat_graph = malin.flatten_vertex_tree();
	for (auto i = 0; i < malin.number_of_vertices(); i++)
	{
		cout << "vertex " << i << endl;
//		flat_graph[i].compressed_edges.iter_elms(i, [&](auto edge){
//			cout << edge << " ";
//		});
//		cout << endl;

		cout << "size of walk-tree vector " << flat_graph[i].compressed_walks.size() << endl;
		int inc = 0;
		for (auto wt = flat_graph[i].compressed_walks.begin(); wt != flat_graph[i].compressed_walks.end(); wt++) // print the walk-trees in chronological order
		{
//			inc++;
			cout << "walk-tree " << inc << endl;
			inc++;
			wt->iter_elms(i, [&](auto enc_triplet){
			  auto pair = pairings::Szudzik<types::Vertex>::unpair(enc_triplet);

			  auto walk_id  = pair.first / config::walk_length;                  // todo: needs floor?
			  auto position = pair.first - (walk_id * config::walk_length); // todo: position here starts from 0. verify this one!
			  auto next_vertex   = pair.second;
			  cout << enc_triplet << " ";
			  cout << "{" << walk_id << ", " << position << ", " << next_vertex << "}" << " " << endl;
			});
			cout << endl;
			cout << "size of walk-tree " << wt->size() << endl;
		}
	}

//// ----------------------------------------------
	cout << "(NEW) WALKS" << endl;
	for (auto i = 0; i < total_vertices * config::walks_per_vertex; i++)
		cout << malin.walk_simple_find(i) << endl;
//// ----------------------------------------------

}