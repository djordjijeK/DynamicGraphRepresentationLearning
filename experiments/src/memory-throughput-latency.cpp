#include <wharfmh.h>

void throughput(commandLine& command_line)
{
    string fname             = string(command_line.getOptionValue("-f", default_file_name));
    bool mmap                = command_line.getOption("-m");
    bool is_symmetric        = command_line.getOption("-s");
    bool compressed          = command_line.getOption("-c");
    size_t n_parts           = command_line.getOptionLongValue("-nparts", 1);

    size_t walks_per_vertex = command_line.getOptionLongValue("-w", config::walks_per_vertex);
    size_t length_of_walks  = command_line.getOptionLongValue("-l", config::walk_length);
    string model            = string(command_line.getOptionValue("-model", "deepwalk"));
    double paramP           = command_line.getOptionDoubleValue("-paramP", config::paramP);
    double paramQ           = command_line.getOptionDoubleValue("-paramQ", config::paramQ);
    string init_strategy    = string(command_line.getOptionValue("-init", "weight"));
    size_t n_trials         = command_line.getOptionLongValue("-trials", 3);
	string determinism      = string(command_line.getOptionValue("-det", "true"));
//    double limit            = 5.5;

    config::walks_per_vertex = walks_per_vertex;
    config::walk_length      = length_of_walks;

    std::cout << "Walks per vertex: " << (int) config::walks_per_vertex << std::endl;
    std::cout << "Walk length: " <<  (int) config::walk_length << std::endl;

    if (model == "deepwalk")
    {
        config::random_walk_model = types::RandomWalkModelType::DEEPWALK;

        std::cout << "Walking model: DEEPWALK" << std::endl;
    }
    else if (model == "node2vec")
    {
        config::random_walk_model = types::RandomWalkModelType::NODE2VEC;
        config::paramP = paramP;
        config::paramQ = paramQ;

        std::cout << "Walking model: NODE2VEC | Params (p,q) = " << "(" << config::paramP << "," << config::paramQ << ")" << std::endl;
    }
    else
    {
        std::cerr << "Unrecognized walking model! Abort" << std::endl;
        std::exit(1);
    }

    if (init_strategy == "burnin")
    {
        config::sampler_init_strategy = types::SamplerInitStartegy::BURNIN;

        std::cout << "Sampler strategy: BURNIN" << std::endl;
    }
    else if (init_strategy == "weight")
    {
        config::sampler_init_strategy = types::SamplerInitStartegy::WEIGHT;

        std::cout << "Sampler strategy: WEIGHT" << std::endl;
    }
    else if (init_strategy == "random")
    {
        config::sampler_init_strategy = types::SamplerInitStartegy::RANDOM;

        std::cout << "Sampler strategy: RANDOM" << std::endl;
    }
    else
    {
        std::cerr << "Unrecognized sampler init strategy" << std::endl;
        std::exit(1);
    }

	if (determinism == "true")
	{
		config::determinism = true;
		cout << "Deterministic mode = " << config::determinism << endl;
	}
	else
	{
		config::determinism = false;
		cout << "Deterministic mode = " << config::determinism << endl;
	}

    size_t n;
    size_t m;
    uintE* offsets;
    uintV* edges;
    std::tie(n, m, offsets, edges) = read_unweighted_graph(fname.c_str(), is_symmetric, mmap);

    dygrl::WharfMH WharfMH = dygrl::WharfMH(n, m, offsets, edges);
    WharfMH.generate_initial_random_walks();
//	// --- add memory measurements here
////	WharfMH.memory_footprint();
//	// ----
//
////	exit(666);
//
//    auto batch_sizes = pbbs::sequence<size_t>(1);
//    batch_sizes[0] = 500;
////    batch_sizes[0] = 5;
////    batch_sizes[1] = 50;
////    batch_sizes[2] = 500;
////    batch_sizes[3] = 5000;
////    batch_sizes[4] = 50000; // up to this value
////    batch_sizes[5] = 500000;
//
//    for (short int i = 0; i < batch_sizes.size(); i++)
//    {
//        timer insert_timer("InsertTimer");
//        timer delete_timer("DeleteTimer");
//
//        graph_update_time_on_insert.reset();
//        walk_update_time_on_insert.reset();
//        graph_update_time_on_delete.reset();
//        walk_update_time_on_delete.reset();
//
//        std::cout << "Batch size = " << 2*batch_sizes[i] << " | ";
//
//        double last_insert_time = 0;
//        double last_delete_time = 0;
//
//        auto latency_insert = pbbs::sequence<double>(n_trials);
//        auto latency_delete = pbbs::sequence<double>(n_trials);
//        auto latency        = pbbs::sequence<double>(n_trials);
//
//        double total_insert_walks_affected = 0;
//        double total_delete_walks_affected = 0;
//
//	    int batch_seed[n_trials];
//	    for (auto i = 0; i < n_trials; i++)
//		    batch_seed[i] = i; // say the seed equals to the #trial
//
//	    for (short int trial = 0; trial < n_trials; trial++)
//	    {
//		    cout << "trial-" << trial << " and batch_seed-" << batch_seed[trial] << endl;
//            size_t graph_size_pow2 = 1 << (pbbs::log2_up(n) - 1);
//            auto edges = utility::generate_batch_of_edges(batch_sizes[i], n, batch_seed[trial], false, false);
//
//            std::cout << edges.second << " ";
//
//            insert_timer.start();
//            auto x = WharfMH.insert_edges_batch(edges.second, edges.first, false, true, graph_size_pow2);
//            insert_timer.stop();
//
//            total_insert_walks_affected += x;
//
//            last_insert_time = walk_update_time_on_insert.get_total() - last_insert_time;
//            latency_insert[trial] = (double) last_insert_time / x;
//
//            delete_timer.start();
//            auto y = WharfMH.delete_edges_batch(edges.second, edges.first, false, true, graph_size_pow2);
//            delete_timer.stop();
//
//            total_delete_walks_affected += y;
//
//            last_delete_time = walk_update_time_on_delete.get_total() - last_delete_time;
//            latency_delete[trial] = (double) last_delete_time / y;
//
//            latency[trial] = (double) (last_insert_time + last_delete_time) / (x + y);
//
////            if (insert_timer.get_total() > 2*limit || delete_timer.get_total() > 2*limit) goto endloop;
//
//            // free edges
//            pbbs::free_array(edges.first);
//        }
//
//        std::cout << std::endl;
//
//        std::cout << "Average insert time = " << insert_timer.get_total() / n_trials << std::endl;
//        std::cout << "Average graph update insert time = " << graph_update_time_on_insert.get_total() / n_trials << std::endl;
//        std::cout << "Average walk update insert time = " << walk_update_time_on_insert.get_total() / n_trials
//        << ", average walk affected = " << total_insert_walks_affected / n_trials << std::endl;
//
//        std::cout << "Average delete time = " << delete_timer.get_total() / n_trials << std::endl;
//        std::cout << "Average graph update delete time = " << graph_update_time_on_delete.get_total() / n_trials << std::endl;
//        std::cout << "Average walk update delete time = " << walk_update_time_on_delete.get_total() / n_trials
//        << ", average walk affected = " << total_delete_walks_affected / n_trials << std::endl;
//
//        std::cout << "Average walk insert latency = { ";
//        for(int i = 0; i < n_trials; i++)
//        {
//            std::cout << latency_insert[i] << " ";
//        }
//        std::cout << "}" << std::endl;
//
//        std::cout << "Average walk delete latency = { ";
//        for(int i = 0; i < n_trials; i++)
//        {
//            std::cout << latency_delete[i] << " ";
//        }
//        std::cout << "}" << std::endl;
//
//        std::cout << "Average walk update latency = { ";
//        for(int i = 0; i < n_trials; i++)
//        {
//            std::cout << latency[i] << " ";
//        }
//        std::cout << "}" << std::endl;
//    }
//
////    endloop:
////    std::cout << "Loop ended" << std::endl;
//
//	// --- add memory measurements here
//	WharfMH.memory_footprint();
//	// ----

	int n_batches = 3; // todo: how many batches per batch size?

	// TODO: Why incorrect numbers when MALIN_DEBUG is off?

	auto batch_sizes = pbbs::sequence<size_t>(1);
	batch_sizes[0] = 5000; //5;
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
//		read_access_MAV.reset();
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

			size_t graph_size_pow2 = 1 << (pbbs::log2_up(n) - 1);
			auto edges = utility::generate_batch_of_edges(batch_sizes[i], n, batch_seed[b], false, false);

			std::cout << edges.second << " ";
//			for (auto i = 0; i < edges.second; i++)
//				cout << "edge-" << i + 1 << " is [" << get<0>(edges.first[i]) << ", " << get<1>(edges.first[i]) << "]" << endl;

cout << "1" << endl;
			insert_timer.start();
			auto x = WharfMH.insert_edges_batch(edges.second, edges.first, b+1, false, true, graph_size_pow2); // pass the batch number as well
			insert_timer.stop();
cout << "10" << endl;

			total_insert_walks_affected += x;

			last_insert_time = walk_update_time_on_insert.get_total() - last_insert_time;
			latency_insert[b] = (double) last_insert_time / x;

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

//		// MAV time
//		std::cout << "Average MAV (we are not deleting obsolete parts) = "
//		          << MAV_time.get_total() / n_batches
//		          << std::endl;
//		// read access time MAV
//		std::cout << "Average Read Access Time MAV = "
//		          << read_access_MAV.get_total() / n_batches
//		          << std::endl;

//		std::cout << "Total MAV (we are not deleting obsolete parts) = " << MAV_time.get_total() << std::endl;
//		std::cout << "Total Read Access Time MAV = " << read_access_MAV.get_total() << std::endl;
		std::cout << "Total walk update insert time = " << walk_update_time_on_insert.get_total() << ", average walk affected = " << total_insert_walks_affected / n_batches << std::endl;
		std::cout << "Total #sampled vertices = " << WharfMH.number_sampled_vertices << std::endl;

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

	// Merge all walks after the bathes
//	timer MergeAll("MergeAllTimer", false);
//	MergeAll.start();
//	malin.merge_walk_trees_all_vertices_parallel(n_batches); // use the parallel merging
//	MergeAll.stop();
//	std::cout << "Merge all the walk-trees time: " << MergeAll.get_total() << std::endl;


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
////			inc++;
//			cout << "walk-tree " << inc << endl;
//			inc++;
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
//			cout << "size of walk-tree " << wt->size() << endl;
//		}
//	}
//
//// ----------------------------------------------
//	cout << "(NEW) WALKS" << endl;
//	for (auto i = 0; i < total_vertices * config::walks_per_vertex; i++)
//		cout << malin.walk_simple_find(i) << endl;
//// ----------------------------------------------
}

int main(int argc, char** argv)
{
    std::cout << " - running throughput-latency experiment with " << num_workers() << " threads" << std::endl;
    commandLine command_line(argc, argv, "");

    throughput(command_line);
}
