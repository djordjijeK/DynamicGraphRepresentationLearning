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
	auto scratchWalks = timer("walksFromScratch", false);
	scratchWalks.start();
    WharfMH.generate_initial_random_walks();
	scratchWalks.stop();
	cout << "Produce the initial walk corpus: " << scratchWalks.get_total() << endl;

	cout << "Keep a backup of initial walk corpus" << endl;
	for (auto& entry : WharfMH.walk_storage.lock_table())
	{
		WharfMH.walk_storage_backup.insert(entry.first, entry.second);
	}
	assert(WharfMH.walk_storage.size() == WharfMH.walk_storage_backup.size());

	int n_batches = 5; // Leave this 5 --> as we do 5 INSERTIONS and 5 DELETIONS

	auto batch_sizes = pbbs::sequence<size_t>(1);
	batch_sizes[0] = 5000; // todo: set the batch size before running the experiment
//	batch_sizes[1] = 50;
//	batch_sizes[2] = 500;
//	batch_sizes[3] = 5000;
//	batch_sizes[4] = 10000;
//	batch_sizes[5] = 15000;
//	batch_sizes[6] = 25000;
//	batch_sizes[7] = 50000;
//  batch_sizes[5] = 500000;

	for (short int i = 0; i < batch_sizes.size(); i++)
	{
		timer insert_timer("InsertTimer");
		timer delete_timer("DeleteTimer");

		graph_update_time_on_insert.reset();
		walk_update_time_on_insert.reset();
		graph_update_time_on_delete.reset();
		walk_update_time_on_delete.reset();

		if (i > 0)
		{
			// Bring back the initial walk corpus
			WharfMH.walk_storage.clear();
			cout << "Bringing back the initial walk corpus..." << endl;
			for (auto& entry : WharfMH.walk_storage_backup.lock_table())
			{
				WharfMH.walk_storage.insert(entry.first, entry.second);
			}
			assert(WharfMH.walk_storage.size() == WharfMH.walk_storage_backup.size());
			cout << "Done...now the walk corpus is the same as the initial." << endl;
		}

		std::cout << "Batch size = " << 2 * batch_sizes[i] << " | ";

		double last_insert_time = 0; double last_delete_time = 0;
		double last_insert_total = 0; double last_delete_total = 0;

		auto latency_insert = pbbs::sequence<double>(n_batches);
		auto latency_delete = pbbs::sequence<double>(n_batches);
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
//cout << "1" << endl;
			// INSERT a generated batch of edges (and update the walks)
			insert_timer.start();
			auto x = WharfMH.insert_edges_batch(edges.second, edges.first, b+1, false, true, graph_size_pow2); // pass the batch number as well
			insert_timer.stop();
			total_insert_walks_affected += x;
//			last_insert_time = walk_update_time_on_insert.get_total() - last_insert_time;
			last_insert_time = walk_update_time_on_insert.get_total() - last_insert_total;
			last_insert_total = walk_update_time_on_insert.get_total();
			latency_insert[b] = (double) last_insert_time / x;
//cout << "2" << endl;

			// DELETE the same generated batch of edges (and update the walks)
			delete_timer.start();     // todo: check the batch number that you pass. REMARK: for baseline it does not matter
			auto y = WharfMH.delete_edges_batch(edges.second, edges.first, b+1, false, true, graph_size_pow2); // pass the batch number as well
			delete_timer.stop();
//cout << "2-1" << endl;
			total_delete_walks_affected += y;
			// last_insert_time = walk_update_time_on_insert.get_total() - last_insert_time;
			last_delete_time  = walk_update_time_on_delete.get_total() - last_delete_total;
			last_delete_total = walk_update_time_on_delete.get_total();
			latency_delete[b] = (double) last_delete_time / y;
//cout << "3" << endl;

//			latency[b] = latency_insert[b];
			latency[b] = (last_insert_time + last_delete_time) / (x + y); // latency of updating one random walk

//cout << "4" << endl;

			// free edges
			pbbs::free_array(edges.first);

			cout << "STATS AT BATCH-" << (b+1) << " INSERTION" << endl;
			std::cout << "Average insert time = " << insert_timer.get_total() / (b+1) << std::endl;
			std::cout << "Average graph update insert time = " << graph_update_time_on_insert.get_total() / (b+1) << std::endl;
			std::cout << "Average walk update insert time = " << walk_update_time_on_insert.get_total() / (b+1) <<
			", average walk affected = " << total_insert_walks_affected / (b+1) << std::endl;
			cout << "walk update time now: " << last_insert_time << endl;
			cout << "---" << endl;
			cout << "STATS AT BATCH-" << (b+1) << " DELETION" << endl;
			std::cout << "Average deletion time = " << delete_timer.get_total() / (b+1) << std::endl;
			std::cout << "Average graph update delete time = " << graph_update_time_on_delete.get_total() / (b+1) << std::endl;
			std::cout << "Average walk update delete time = "  << walk_update_time_on_delete.get_total() / (b+1) <<
			", average walk affected = " << total_delete_walks_affected / (b+1) << std::endl;
			cout << "walk update time now: " << last_delete_time << endl;
			cout << "---" << endl;
			cout << "---" << endl;
//cout << "5" << endl;
		}
		cout << fixed;
		std::cout << std::endl;

		cout << "STATS FOR ALL BATCHES" << endl;
		std::cout << "Average insert time = " << insert_timer.get_total() / n_batches << std::endl;
		std::cout << "Average graph update insert time = " << graph_update_time_on_insert.get_total() / n_batches << std::endl;
		std::cout << "Average walk update insert time = " << walk_update_time_on_insert.get_total() / n_batches << ", average walk affected = " << total_insert_walks_affected / n_batches << std::endl;

		std::cout << "Average delete time = " << delete_timer.get_total() / n_batches << std::endl;
		std::cout << "Average graph update delete time = " << graph_update_time_on_delete.get_total() / n_batches << std::endl;
		std::cout << "Average walk update delete time = " << walk_update_time_on_delete.get_total() / n_batches << ", average walk affected = " << total_delete_walks_affected / n_batches << std::endl;

		std::cout << "Total walk update insert time = " << walk_update_time_on_insert.get_total() << ", average walk affected = " << total_insert_walks_affected / n_batches << std::endl;
		std::cout << "Total walk update delete time = " << walk_update_time_on_delete.get_total() << ", average walk affected = " << total_delete_walks_affected / n_batches << std::endl;
		std::cout << "Total #sampled vertices = " << WharfMH.number_sampled_vertices << std::endl;

		// latencies
		std::cout << "Average walk insert latency = { ";
		for (int i = 0; i < n_batches; i++) {
			std::cout << latency_insert[i] << " ";
		}
		std::cout << "}" << std::endl;

		std::cout << "Average walk delete latency = { ";
		for (int i = 0; i < n_batches; i++) {
			std::cout << latency_delete[i] << " ";
		}
		std::cout << "}" << std::endl;

		double total_latency = 0.0;
		std::cout << "Average walk total latency = { ";
		for (int i = 0; i < n_batches; i++) {
			std::cout << latency[i] << " ";
			total_latency += latency[i];
		}
		std::cout << "}" << std::endl;

//		cout << "(1) throughput: " << fixed << setprecision(8) << total_insert_walks_affected / (walk_update_time_on_insert.get_total() * 1.0) << endl;
		cout << "(1) throughput: " << fixed << setprecision(8) <<
			(total_insert_walks_affected + total_delete_walks_affected) / (walk_update_time_on_insert.get_total() + walk_update_time_on_delete.get_total()) << endl;
		cout << "(2) average latency: " << fixed << setprecision(8) << total_latency / n_batches << endl;
	}

	// Measure read walk time
/*	auto ReadWalks = timer("ReadWalks", false);
	ReadWalks.start();
	for (auto i = 0; i < WharfMH.number_of_vertices() * config::walks_per_vertex; i++)
		WharfMH.walk_silent(i);
//		cout << WharfMH.walk(i) << endl;
	ReadWalks.stop();
	cout << "Read all walks time: " << ReadWalks.get_total() << endl;*/

	// Measure the total memory in the end
	WharfMH.memory_footprint();

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
