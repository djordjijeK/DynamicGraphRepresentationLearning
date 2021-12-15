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
    int n_trials = 3;

    double limit = 5.5;

//		WharfMH.walk_cout(1);
//		cout << WharfMH.walk(1);
//		WharfMH.walk_cout(13);

		cout << "WALKS" << endl;
		for (auto i = 0; i < total_vertices * config::walks_per_vertex; i++)
				cout << WharfMH.walk(i) << endl;

		cout << "INV INDEX" << endl;
		WharfMH.walk_index_print();


		exit(1);


    auto batch_sizes = pbbs::sequence<size_t>(1);
    batch_sizes[0] = 5;
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
            auto edges = utility::generate_batch_of_edges(batch_sizes[i], total_vertices, false, false);

            std::cout << edges.second << " ";

            insert_timer.start();
            auto x = WharfMH.insert_edges_batch(edges.second, edges.first, false, true, graph_size_pow2);
            insert_timer.stop();

            total_insert_walks_affected += x;

            last_insert_time = walk_update_time_on_insert.get_total() - last_insert_time;
            latency_insert[trial] = (double) last_insert_time / x;

            delete_timer.start();
            auto y = WharfMH.delete_edges_batch(edges.second, edges.first, false, true, graph_size_pow2);
            delete_timer.stop();

            total_delete_walks_affected += y;

            last_delete_time = walk_update_time_on_delete.get_total() - last_delete_time;
            latency_delete[trial] = (double) last_delete_time / y;

            latency[trial] = (double) (last_insert_time + last_delete_time) / (x + y);

            if (insert_timer.get_total() > 2*limit || delete_timer.get_total() > 2*limit) goto endloop;

            // free edges
            pbbs::free_array(edges.first);
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

    endloop:
        std::cout << "Loop ended" << std::endl;
}