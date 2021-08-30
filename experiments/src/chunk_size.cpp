#include <malin.h>

void chunk_size(commandLine& command_line)
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

    string determinism      = string(command_line.getOptionValue("-d", "false"));
    string range_search     = string(command_line.getOptionValue("-rs", "true"));
    int head_frequency      = command_line.getOptionIntValue("-hf", 8); // chunk size is 2^8 = 256 (slightly bigger than 128. aspen's machine cachelinesizes)

    config::walks_per_vertex = walks_per_vertex;
    config::walk_length      = length_of_walks;

    std::cout << "Walks per vertex: " << (int) config::walks_per_vertex << std::endl;
    std::cout << "Walk length: " << (int) config::walk_length << std::endl;

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

    // Set up the range search mode ------
    if (range_search == "true")
        config::range_search_mode = true;
    else
        config::range_search_mode = false;

    // Set up the deterministic mode
    if (determinism == "true")
        config::deterministic_mode = true;
    else
        config::deterministic_mode = false;
    // ------------------------------------

    // Assign the head frequency we read
    compressed_lists::head_frequency = (size_t) head_frequency;
    cout << endl << "Head frequency is " << compressed_lists::head_frequency << ", and thus, chunk size is " << (1 << compressed_lists::head_frequency) << endl;
    // ---------------------------------

    size_t n;
    size_t m;
    uintE* offsets;
    uintV* edges;
    std::tie(n, m, offsets, edges) = read_unweighted_graph(fname.c_str(), is_symmetric, mmap);

    dygrl::Malin malin = dygrl::Malin(n, m, offsets, edges, false);
    dygrl::Malin malin_2 = dygrl::Malin(n, m, offsets, edges, true);

    // -------------------------------------------------------------------
    // Produce initial walks corpus with all threads -- full parallel mode
    auto fully_parallel_timer = timer("WalkCorpusParallel", false); fully_parallel_timer.start();
    malin.generate_initial_random_walks();
    auto corpus_parallel_time =  fully_parallel_timer.get_total();
    // -------------------------------------------------------------------

    // -------------------------------------------------------------------
    // Produce initial walks corpus with one thread -- one thread mode
    auto single_thread_timer = timer("WalkCorpusSingleThread", false); single_thread_timer.start();
    malin_2.generate_initial_random_walks_sequential();
    auto corpus_single_thread_time =  single_thread_timer.get_total();
    // -------------------------------------------------------------------

    // Print the memory requirements to store the produced walk corpus
//    malin.memory_footprint();

//    auto traverse_walk_corpus = timer("WalkCorpusTraversal", false); traverse_walk_corpus.start();
//    // print random walks
//    cout << "k=" << compressed_lists::head_frequency << " walk corpus traversal started..." << endl;
//    for(int i = 0; i < config::walks_per_vertex * malin.number_of_vertices(); i++)
//        malin.traverse_walk(i) ;
//    auto walk_corpus_traversal_time = traverse_walk_corpus.get_total();
//    cout << "k=" << compressed_lists::head_frequency << " walk corpus traversal finished..." << endl;

    cout << "Time to produce the walk  corpus (fully parallel): " << corpus_parallel_time << endl
         << "Time to produce the walk corpus   (single thread): " << corpus_single_thread_time << endl
         << "Self speed-up (SU): " << corpus_single_thread_time / corpus_parallel_time << endl;

}


int main(int argc, char** argv)
{
    std::cout << "Running experiment with: " << num_workers() << " threads." << std::endl;
    commandLine command_line(argc, argv, "");

    chunk_size(command_line);
}



