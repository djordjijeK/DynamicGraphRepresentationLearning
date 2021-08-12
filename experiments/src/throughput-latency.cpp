#include <malin.h>

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
        config::sampler_init_strategy = types::SamplerInitStrategy::BURNIN;

        std::cout << "Sampler strategy: BURNIN" << std::endl;
    }
    else if (init_strategy == "weight")
    {
        config::sampler_init_strategy = types::SamplerInitStrategy::WEIGHT;

        std::cout << "Sampler strategy: WEIGHT" << std::endl;
    }
    else if (init_strategy == "random")
    {
        config::sampler_init_strategy = types::SamplerInitStrategy::RANDOM;

        std::cout << "Sampler strategy: RANDOM" << std::endl;
    }
    else
    {
        std::cerr << "Unrecognized sampler init strategy" << std::endl;
        std::exit(1);
    }

    size_t n;
    size_t m;
    uintE* offsets;
    uintV* edges;
    std::tie(n, m, offsets, edges) = read_unweighted_graph(fname.c_str(), is_symmetric, mmap);

    dygrl::Malin malin = dygrl::Malin(n, m, offsets, edges);
    malin.generate_initial_random_walks();

    auto batch_sizes = pbbs::sequence<size_t>(6);
    batch_sizes[0] = 5;
    batch_sizes[1] = 50;
    batch_sizes[2] = 500;
    batch_sizes[3] = 5000;
    batch_sizes[4] = 50000;
    batch_sizes[5] = 500000;

    for (short int i = 0; i < batch_sizes.size(); i++)
    {
        timer insert_timer("InsertTimer", false);
        timer delete_timer("DeleteTimer", false);

        graph_update_time_on_insert.reset();
        walk_update_time_on_insert.reset();
        graph_update_time_on_delete.reset();
        walk_update_time_on_delete.reset();

        std::cout << std::endl;
        std::cout << "Batch size = " << 2 * batch_sizes[i] << " | ";

        double last_insert_time = 0;
        double last_delete_time = 0;

        auto latency_insert = pbbs::sequence<double>(n_trials);
        auto latency_delete = pbbs::sequence<double>(n_trials);
        auto latency        = pbbs::sequence<double>(n_trials);

        double total_insert_walks_affected = 0;
        double total_delete_walks_affected = 0;

        for (short int trial = 0; trial < n_trials; trial++)
        {
            size_t graph_size_pow2 = 1 << (pbbs::log2_up(n) - 1);
            auto edges = utility::generate_batch_of_edges(batch_sizes[i], n, false, false);

            std::cout << edges.second << " ";

            insert_timer.start();
            auto x = malin.insert_edges_batch(edges.second, edges.first, false, true, graph_size_pow2);
            insert_timer.stop();

            total_insert_walks_affected += x.size();

            last_insert_time = walk_update_time_on_insert.get_total() - last_insert_time;
            latency_insert[trial] = last_insert_time / x.size();

            delete_timer.start();
            auto y = malin.delete_edges_batch(edges.second, edges.first, false, true, graph_size_pow2);
            delete_timer.stop();

            total_delete_walks_affected += y.size();

            last_delete_time = walk_update_time_on_delete.get_total() - last_delete_time;
            latency_delete[trial] = last_delete_time / y.size();

            latency[trial] = (last_insert_time + last_delete_time) / (x.size() + y.size());

            // free edges
            pbbs::free_array(edges.first);
        }

        std::cout << std::endl;

        std::cout << "Average insert time = " << insert_timer.get_total() / n_trials << std::endl;
        std::cout << "Average graph update insert time = " << graph_update_time_on_insert.get_total() / n_trials << std::endl;
        std::cout << "Average walk update insert time = " << walk_update_time_on_insert.get_total() / n_trials << " | Average number of walks affected = " << total_insert_walks_affected / n_trials << std::endl;

        std::cout << "Average delete time = " << delete_timer.get_total() / n_trials << std::endl;
        std::cout << "Average graph update delete time = " << graph_update_time_on_delete.get_total() / n_trials << std::endl;
        std::cout << "Average walk update delete time = " << walk_update_time_on_delete.get_total() / n_trials << " | Average number of walks affected = " << total_delete_walks_affected / n_trials << std::endl;

        std::cout << "Average walk insert latency = { ";
        for (int i = 0; i < n_trials; i++) {
            std::cout << latency_insert[i] << " ";
        }
        std::cout << "}" << std::endl;

        std::cout << "Average walk delete latency = { ";
        for (int i = 0; i < n_trials; i++) {
            std::cout << latency_delete[i] << " ";
        }
        std::cout << "}" << std::endl;

        std::cout << "Average walk update latency = { ";
        for (int i = 0; i < n_trials; i++) {
            std::cout << latency[i] << " ";
        }
        std::cout << "}" << std::endl;
    }

    malin.destroy_index();
    timer generate_initial_walks("Generate Initial Random Walks", false);
    for (int i = 0; i < n_trials; i++)
    {
        generate_initial_walks.start();
        malin.generate_initial_random_walks();
        generate_initial_walks.stop();

        malin.destroy_index();
    }

    std::cout << std::endl
              << "Average time to generate random walks from scratch = "
              << generate_initial_walks.get_total() / n_trials << std::endl;

    std::cout << std::endl;
}

int main(int argc, char** argv)
{
    std::cout << " - running throughput-latency experiment with " << num_workers() << " threads" << std::endl;
    commandLine command_line(argc, argv, "");

    throughput(command_line);
}

