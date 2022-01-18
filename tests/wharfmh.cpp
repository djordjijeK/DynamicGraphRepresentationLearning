#include <gtest/gtest.h>

#include <wharfmh.h>

class WharfMHTest : public testing::Test
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
        std::string default_file_path = "data/aspen-paper-graph";
//        std::string default_file_path = "data/email-graph";
//        std::string default_file_path = "data/flickr-graph";
};

void WharfMHTest::SetUp()
{
    std::cout << "-----------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "WharfMH running with " << num_workers() << " threads" << std::endl;

    // transform an input graph file into an adjacency graph format
    std::string command = "./SNAPtoAdj -s -f " + this->default_file_path + " data/adjacency-graph-format.txt";
    int result = system(command.c_str());

    if (result != 0)
    {
        std::cerr << "WharfMHTest::SetUp::Input file could not be transformed!" << std::endl;
        exit(1);
    }

    std::tie(total_vertices, total_edges, offsets, edges) = read_unweighted_graph("data/adjacency-graph-format.txt", is_symmetric, mmap);
    std::cout << std::endl;
}

void WharfMHTest::TearDown()
{
    // remove adjaceny graph format representation
    int graph = system("rm -rf data/adjacency-graph-format.txt");

    if (graph != 0)
    {
        std::cerr << "WharfMHTest::TearDown::Could not remove static graph input file" << std::endl;
    }

    std::cout << "-----------------------------------------------------------------------------------------------------" << std::endl;
}

TEST_F(WharfMHTest, WharfMHThroughputLatency)
{
    dygrl::WharfMH WharfMH = dygrl::WharfMH(total_vertices, total_edges, offsets, edges);
    WharfMH.generate_initial_random_walks();
    int n_trials = 1; //3;

    double limit = 5.5;

//		WharfMH.walk_cout(1);
//		cout << WharfMH.walk(1);    
//		WharfMH.walk_cout(13);

// ----------------------------------------------
//		cout << "WALKS" << endl;
//		for (auto i = 0; i < total_vertices * config::walks_per_vertex; i++)
//				cout << WharfMH.walk(i) << endl;
//
//		cout << "INV INDEX" << endl;
//		WharfMH.walk_index_print();
// ----------------------------------------------


//		exit(1);


    auto batch_sizes = pbbs::sequence<size_t>(1);
    batch_sizes[0] = 1000; //5;
//    batch_sizes[1] = 50;
//    batch_sizes[2] = 500;
//    batch_sizes[3] = 5000;
//    batch_sizes[4] = 50000;
//    batch_sizes[5] = 500000;

    for (short int i = 0; i < batch_sizes.size(); i++)
    {
        timer insert_timer("InsertTimer");
        timer delete_timer("DeleteTimer");

        graph_update_time_on_insert.reset();
        walk_update_time_on_insert.reset();
        graph_update_time_on_delete.reset();
        walk_update_time_on_delete.reset();

        std::cout << "Batch size = " << 2*batch_sizes[i] << " | ";

        double last_insert_time = 0;
        double last_delete_time = 0;

        auto latency_insert = pbbs::sequence<double>(n_trials);
        auto latency_delete = pbbs::sequence<double>(n_trials);
        auto latency        = pbbs::sequence<double>(n_trials);

        double total_insert_walks_affected = 0;
        double total_delete_walks_affected = 0;

        for (short int trial = 0; trial < n_trials; trial++)
        {
            size_t graph_size_pow2 = 1 << (pbbs::log2_up(total_vertices) - 1);
	        cout << "graph size pow: " << graph_size_pow2 << endl;
	        cout << "batch size: " << batch_sizes[i] << endl;
	        cout << "total V: " << total_vertices << endl;
            auto edges = utility::generate_batch_of_edges(batch_sizes[i], total_vertices, false, false);
	        // ---
	        for (auto i = 0; i < edges.second; i++)
		        cout << "edge-" << i + 1 << " is [" << get<0>(edges.first[i]) << ", " << get<1>(edges.first[i]) << "]" << endl;

//			pair<tuple<unsigned int, unsigned int>*, unsigned long> edges = {{2, 4}, 1};

			// ----
			// Print the edges that you generated
//			cout << "edges generated are..." << endl;
//			for (auto i = 0; i < edges.second; i++)
//			{
//				cout << get<0>(edges.first[i]) << "," << get<1>(edges.first[i]) << endl;
//			}
			// ----

            std::cout << edges.second << " ";

            insert_timer.start();
            auto x = WharfMH.insert_edges_batch(edges.second, edges.first, false, true, graph_size_pow2);
            insert_timer.stop();

            total_insert_walks_affected += x;

            last_insert_time = walk_update_time_on_insert.get_total() - last_insert_time;
            latency_insert[trial] = (double) last_insert_time / x;

//            delete_timer.start();
//            auto y = WharfMH.delete_edges_batch(edges.second, edges.first, false, true, graph_size_pow2);
//            delete_timer.stop();
//
//            total_delete_walks_affected += y;
//
//            last_delete_time = walk_update_time_on_delete.get_total() - last_delete_time;
//            latency_delete[trial] = (double) last_delete_time / y;

//            latency[trial] = (double) (last_insert_time + last_delete_time) / (x + y);
	        latency[trial] = latency_insert[trial]; // todo: for now latency insert trial only

//            if (insert_timer.get_total() > 2*limit || delete_timer.get_total() > 2*limit) goto endloop;

            // free edges
            pbbs::free_array(edges.first);
        }

        std::cout << std::endl;

        std::cout << "Average insert time = " << insert_timer.get_total() / n_trials << std::endl;
        std::cout << "Average graph update insert time = " << graph_update_time_on_insert.get_total() / n_trials << std::endl;
        std::cout << "Average walk update insert time = " << walk_update_time_on_insert.get_total() / n_trials
                  << ", average walk affected = " << total_insert_walks_affected / n_trials << std::endl;

//        std::cout << "Average delete time = " << delete_timer.get_total() / n_trials << std::endl;
//        std::cout << "Average graph update delete time = " << graph_update_time_on_delete.get_total() / n_trials << std::endl;
//        std::cout << "Average walk update delete time = " << walk_update_time_on_delete.get_total() / n_trials
//                  << ", average walk affected = " << total_delete_walks_affected / n_trials << std::endl;

        std::cout << "Average walk insert latency = { ";
        for(int i = 0; i < n_trials; i++)
        {
            std::cout << latency_insert[i] << " ";
        }
        std::cout << "}" << std::endl;

//        std::cout << "Average walk delete latency = { ";
//        for(int i = 0; i < n_trials; i++)
//        {
//            std::cout << latency_delete[i] << " ";
//        }
//        std::cout << "}" << std::endl;

        std::cout << "Average walk update latency = { ";
        for(int i = 0; i < n_trials; i++)
        {
            std::cout << latency[i] << " ";
        }
        std::cout << "}" << std::endl;
    }

// ----------------------------------------------
//	cout << "(NEW) WALKS" << endl;
//	for (auto i = 0; i < total_vertices * config::walks_per_vertex; i++)
//		cout << WharfMH.walk(i) << endl;
//
//	cout << "(NEW) INV INDEX" << endl;
//	WharfMH.walk_index_print();
// ----------------------------------------------

    endloop:
        std::cout << "Loop ended" << std::endl;
}

TEST_F(WharfMHTest, BatchGenerator)
{
	cout << "first batch" << endl;
	auto edges = utility::generate_batch_of_edges(10, 6, false, false);
	for (auto i = 0; i < edges.second; i++)
		cout << "edge-" << i + 1 << " is [" << get<0>(edges.first[i]) << ", " << get<1>(edges.first[i]) << "]" << endl;

	cout << "second batch" << endl;
	edges = utility::generate_batch_of_edges(10, 6, 5, false, false);
	for (auto i = 0; i < edges.second; i++)
		cout << "edge-" << i + 1 << " is [" << get<0>(edges.first[i]) << ", " << get<1>(edges.first[i]) << "]" << endl;

	cout << "third batch" << endl;
	edges = utility::generate_batch_of_edges(10, 6, 7, false, false);
	for (auto i = 0; i < edges.second; i++)
		cout << "edge-" << i + 1 << " is [" << get<0>(edges.first[i]) << ", " << get<1>(edges.first[i]) << "]" << endl;
}

TEST_F(WharfMHTest, WharfInsertOnlyWorkload) {
	dygrl::WharfMH malin = dygrl::WharfMH(total_vertices, total_edges, offsets, edges);
	malin.generate_initial_random_walks();
	int n_batches = 3; // todo: how many batches per batch size?

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
//		// --- profiling initialization
//		walk_insert_init.reset();
//		walk_insert_2jobs.reset();
//		walk_insert_2accs.reset();
//		ij.reset();
//		dj.reset();
//		walk_find_in_vertex_tree.reset();
//		walk_find_next_tree.reset();
//		szudzik_hash.reset();
//		fnir_tree_search.reset();
//		MAV_time.reset();
//		// ---

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
			for (auto i = 0; i < edges.second; i++)
				cout << "edge-" << i + 1 << " is [" << get<0>(edges.first[i]) << ", " << get<1>(edges.first[i]) << "]" << endl;

			insert_timer.start();
			auto x = malin.insert_edges_batch(edges.second, edges.first, b+1, false, true, graph_size_pow2); // pass the batch number as well
			insert_timer.stop();

			total_insert_walks_affected += x;

			last_insert_time = walk_update_time_on_insert.get_total() - last_insert_time;
			latency_insert[b] = (double) last_insert_time / x;

			latency[b] = latency_insert[b];

			// free edges
			pbbs::free_array(edges.first);
		}

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

//		std::cout << "Total MAV (we are not deleting obsolete parts) = " << MAV_time.get_total() << std::endl;
//		std::cout << "Total Read Access Time MAV = " << read_access_MAV.get_total() << std::endl;
		std::cout << "Total walk update insert time = " << walk_update_time_on_insert.get_total() << ", average walk affected = " << total_insert_walks_affected / n_batches << std::endl;

//		// MAV time
//		std::cout << "Average MAV (we are not deleting obsolete parts) = "
//		          << MAV_time.get_total() / n_batches
//		          << std::endl;

//		// --- profiling ---
//		std::cout << "{ total profiling for insert and delete" << std::endl;
//		std::cout << "Initialization: "
//		          << walk_insert_init.get_total() / n_batches << " ("
//		          << (walk_insert_init.get_total() * 100) /
//		             (walk_insert_init.get_total() +
//		              walk_insert_2jobs.get_total() +
//		              walk_insert_2accs.get_total()) << "%)" << std::endl;
//		std::cout << "Insert/Delete Jobs: "
//		          << walk_insert_2jobs.get_total() / n_batches << " ("
//		          << (walk_insert_2jobs.get_total() * 100) /
//		             (walk_insert_init.get_total() +
//		              walk_insert_2jobs.get_total() +
//		              walk_insert_2accs.get_total()) << "%)" << std::endl;
//		std::cout << "InsertJob: " << ij.get_total() / n_batches
//		          << " | DeleteJob: " << dj.get_total() / n_batches << std::endl;
//		std::cout << "FindInVertexTree in DeleteJob total: "
//		          << walk_find_in_vertex_tree.get_total() / n_batches
//		          << std::endl;
//		std::cout << "FindNext in DeleteJob total: "
//		          << walk_find_next_tree.get_total() / n_batches << std::endl;
//		std::cout << "FindNext (search of the tree): "
//		          << fnir_tree_search.get_total() / n_batches << std::endl;
//		std::cout << "Sudzik total: " << szudzik_hash.get_total() / n_batches
//		          << std::endl;
//
//		std::cout << "Accumulators: "
//		          << walk_insert_2accs.get_total() / n_batches << " ("
//		          << (walk_insert_2accs.get_total() * 100) /
//		             (walk_insert_init.get_total() +
//		              walk_insert_2jobs.get_total() +
//		              walk_insert_2accs.get_total()) << "%)" << std::endl;
//		std::cout << "}" << std::endl;
//		// --- profiling ---

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

	std::cout << "Total #sampled vertices = " << malin.number_sampled_vertices << std::endl;

//// ----------------------------------------------
	cout << "(NEW) WALKS" << endl;
	for (auto i = 0; i < total_vertices * config::walks_per_vertex; i++)
		cout << malin.walk(i) << endl;

	cout << "(NEW) INV INDEX" << endl;
	malin.walk_index_print();
//// ----------------------------------------------

//	auto flat_graph = malin.flatten_vertex_tree();
//	for (auto i = 0; i < malin.number_of_vertices(); i++)
//	{
//		cout << "vertex " << i << endl;
////		flat_graph[i].compressed_edges.iter_elms(i, [&](auto edge){
////			cout << edge << " ";
////		});
////		cout << endl;
//
//		cout << "size of walk-tree vector " << flat_graph[i].compressed_walks.size() << endl;
//		int inc = 0;
//		for (auto wt = flat_graph[i].compressed_walks.begin(); wt != flat_graph[i].compressed_walks.end(); wt++) // print the walk-trees in chronological order
//		{
//			inc++;
//			cout << "walk-tree " << inc << endl;
//			wt->iter_elms(i, [&](auto enc_triplet){
//			  auto pair = pairings::Szudzik<types::Vertex>::unpair(enc_triplet);
//
//			  auto walk_id  = pair.first / config::walk_length;                  // todo: needs floor?
//			  auto position = pair.first - (walk_id * config::walk_length); // todo: position here starts from 0. verify this one!
//			  auto next_vertex   = pair.second;
////				cout << enc_triplet << " ";
////			  cout << "{" << walk_id << ", " << position << ", " << next_vertex << "}" << " " << endl;
//			});
//			cout << endl;
//		}
//	}
}