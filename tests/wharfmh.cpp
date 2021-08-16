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
        bool mmap = false;          // TODO @Djordjije: do we need this?
        bool is_symmetric = true;   // TODO @Djordjije: do we need this?
        std::string default_file_path = "data/email-graph";
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

TEST_F(WharfMHTest, WharfMHConstructor)
{
    dygrl::WharfMH WharfMH = dygrl::WharfMH(total_vertices, total_edges, offsets, edges, false);

    // assert the number of vertices and edges in a graph
    ASSERT_EQ(WharfMH.number_of_vertices(), total_vertices);
    ASSERT_EQ(WharfMH.number_of_edges(), total_edges);

    // construct a flat snapshot of a graph
    auto flat_snapshot = WharfMH.flatten_vertex_tree();

    // assert
    parallel_for(0, total_vertices, [&] (long i)
    {
        size_t off = offsets[i];
        size_t degree = ((i == (total_vertices - 1)) ? total_edges : offsets[i+1]) - off;
        auto S = pbbs::delayed_seq<uintV>(degree, [&] (size_t j) { return edges[off + j]; });

        // assert expected degrees
        ASSERT_EQ(flat_snapshot[i].compressed_edges.degree(), degree);

        auto edges = flat_snapshot[i].compressed_edges.get_edges(i);

        // assert expected neighbours
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

TEST_F(WharfMHTest, WharfMHDestructor)
{
    dygrl::WharfMH WharfMH = dygrl::WharfMH(total_vertices, total_edges, offsets, edges);

    WharfMH.print_memory_pool_stats();
    WharfMH.destroy();
    WharfMH.print_memory_pool_stats();

    // assert vertices and edges
    ASSERT_EQ(WharfMH.number_of_vertices(), 0);
    ASSERT_EQ(WharfMH.number_of_edges(), 0);

    // construct a flat snapshot of a graph
    auto flat_snapshot = WharfMH.flatten_vertex_tree();

    // assert that flat snapshot does not exits
    ASSERT_EQ(flat_snapshot.size(), 0);
}

TEST_F(WharfMHTest, InsertBatchOfEdges)
{
    // create wharf instance (vertices & edges)
    dygrl::WharfMH WharfMH = dygrl::WharfMH(total_vertices, total_edges, offsets, edges);
    auto start_edges = WharfMH.number_of_edges();

    // geneate edges
//    auto edges = utility::generate_batch_of_edges(2 * WharfMH.number_of_vertices(), WharfMH.number_of_vertices(), false, false);

    std::tuple<uintV, uintV>* generated_edges = new std::tuple<uintV, uintV>[1];
    generated_edges[0] = {1, 3};
    auto edges_generated = 1;

    // insert batch of edges
    WharfMH.insert_edges_batch(edges_generated, generated_edges, true, false);

    std::cout << "Edges before batch insert: " << start_edges << std::endl;
    std::cout << "Edges after batch insert: "  << WharfMH.number_of_edges() << std::endl;

    // assert edge insertion
    ASSERT_GE(WharfMH.number_of_edges(), start_edges);
}

TEST_F(WharfMHTest, DeleteBatchOfEdges)
{
    // create wharf instance (vertices & edges)
    dygrl::WharfMH WharfMH = dygrl::WharfMH(total_vertices, total_edges, offsets, edges);
    auto start_edges = WharfMH.number_of_edges();

    // geneate edges
//    auto edges = utility::generate_batch_of_edges(2 * WharfMH.number_of_vertices(), WharfMH.number_of_vertices(), false, false);

    std::tuple<uintV, uintV>* generated_edges = new std::tuple<uintV, uintV>[1];
    generated_edges[0] = {2, 5};
    auto edges_generated = 1;

    // insert batch of edges
    WharfMH.delete_edges_batch(edges_generated, generated_edges, true, false);

    std::cout << "Edges before batch delete: " << start_edges << std::endl;
    std::cout << "Edges after batch delete: " << WharfMH.number_of_edges() << std::endl;

    // assert edge deletion
    ASSERT_LE(WharfMH.number_of_edges(), start_edges);
}

TEST_F(WharfMHTest, DEV)
{
    dygrl::WharfMH WharfMH = dygrl::WharfMH(total_vertices, total_edges, offsets, edges);
    WharfMH.create_random_walks();

    for(int i = 0; i < config::walks_per_vertex * WharfMH.number_of_vertices(); i++)
    {
        std::cout << WharfMH.rewalk(i) << std::endl;
    }

    std::tuple<uintV, uintV>* generated_edges = new std::tuple<uintV, uintV>[2];
    generated_edges[0] = {1, 3};
    generated_edges[1] = {3, 1};
    auto edges_generated = 1;

    // insert batch of edges
    WharfMH.insert_edges_batch(edges_generated, generated_edges, true, false);
//    WharfMH.delete_edges_batch(edges_generated, generated_edges, true, false);

    for(int i = 0; i < config::walks_per_vertex * WharfMH.number_of_vertices(); i++)
    {
        std::cout << WharfMH.rewalk(i) << std::endl;
    }
}

