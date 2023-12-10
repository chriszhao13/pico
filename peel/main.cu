#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include "./inc/gpu_memory_allocation.h"
#include "./src/buffer.cc"
#include "./src/peel-dyn.cc"
#include "./src/peelone-dyn.cc"


void PrintMax(uint *result, uint n)
{

    uint max = 0;
    uint min = 10;
    for (int i = 0; i < n; i++)
    {

        if (max < result[i])
            max = result[i];
        if (min > result[i])
            min = result[i];
    }

    cout << " Maxdegree = " << max << endl;
    //cout << " Min K = " << min << endl;
}

template <class T>
void repSimulation(int (*kern)(T), Graph &g)
{
    float sum = 0;

    int rep = 10; // number of iterations...
    uint max_time = 0;
    uint min_time = UINT_MAX;
    assert(rep >= 3);
    for (int i = 0; i < rep; i++)
    {
        unsigned int t = (*kern)(g);
        cout << t << "ms ";
        if (max_time < t)
            max_time = t;
        if (min_time > t)
            min_time = t;
        sum += t;
    }
    sum = sum - max_time - min_time;
    cout << " ave: " << sum * 1.0 / (rep - 2) << endl;
}

void STDdegrees(Graph &g)
{
    double sum = std::accumulate(g.degrees, g.degrees + g.V, 0.0);
    double mean = sum / g.V;

    std::vector<double> diff(g.V);
    std::transform(g.degrees, g.degrees + g.V, diff.begin(),
                   std::bind2nd(std::minus<double>(), mean));
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / g.V);
    cout<<endl<<"STD: "<<endl;
    cout << stdev << endl;
}

int main(int argc, char *argv[])
{

    if (argc < 2)
    {
        cout << "./kcore file deviceID Repeattime" << endl;
        exit(-1);
    }

    std::string ds = argv[1];

    // cudaSetDevice(std::atoi(argv[2]));

    // int rep = std::atoi(argv[3]);

    cudaFree(0);
    cout << "Graph loading Started... " << endl;
    Graph g(ds);

    cout << "V: " << g.V << endl;
    cout << "E: " << g.E / 2 << endl;

    STDdegrees(g);
    PrintMax(g.degrees, g.V);

    cout<<"ICDE23_peel: ";
    repSimulation(peel_dyn, g);
    cout<<"Kmax: "<<g.kmax<<endl;

    cout<<"Our_Paper_peel: ";
    repSimulation(peelone_dyn, g);
    cout<<"Kmax: "<<g.kmax<<endl;
    return 0;
}
