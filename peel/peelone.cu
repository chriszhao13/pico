//
// Created by Chris on 2022/11/30.
//
#include "shared/graph.h"
#include "shared/Timer.h"
#include "shared/util.h"
#include "shared/virtual_graph.h"
#include "shared/globals.h"
#include "shared/argument.h"
#include "shared/gpu_error.cuh"

// __device__ int DEVICE_atomicsub = 0;

__global__ void kernel(
        uint partSize,

        bool *sighed,

        unsigned int numParts,

        unsigned int *nodePointer,

        PartPointer *partNodePointer,

        unsigned int *edgeList,

        unsigned int *dist,

        bool *finished,

        bool *plus,

        unsigned int level) {

    unsigned int partId = blockDim.x * blockIdx.x + threadIdx.x;

    if (partId < numParts) {

        unsigned int id = partNodePointer[partId].node;
        unsigned int part = partNodePointer[partId].part;

        if (sighed[partId]) {
            return;
        }

        *finished = false;

        if (dist[id] != level) {
            return;
        }

        *plus = false;
        sighed[partId] = true;

        unsigned int thisPointer = nodePointer[id];
        unsigned int degree = edgeList[thisPointer];

        int numParts;


        if (degree % partSize == 0)

            numParts = degree / partSize;

        else

            numParts = degree / partSize + 1;

        unsigned int end;

        unsigned int ofs = thisPointer + part + 1;

        for (int i = 0; i < partSize; i++) {

            if (part + i * numParts >= degree)
                break;

            end = ofs + i * numParts;

            if (dist[edgeList[end]] > level) {

                if (atomicSub(dist + edgeList[end], 1) == level) {

                    dist[edgeList[end]] = level;
                }
            }
        }

    }

}



int main(int argc, char **argv) {

    argument arguments(argc, argv, false, false);

    Graph graph(arguments.input, false);

    graph.ReadGraph();

    VirtualGraph vGraph(graph, arguments.Part_Size);

    vGraph.MakeUGraph();

    uint Part_Size = arguments.Part_Size;

    uint num_nodes = graph.num_nodes;
    uint num_edges = 2 * graph.num_edges;

    cudaSetDevice(arguments.deviceID);
    cudaFree(0);

    uint *dist = new uint[num_nodes];

    for (int i = 0; i < num_nodes; i++) {
        dist[i] = vGraph.outDegree[i];
    }
    unsigned int *d_nodePointer;
    unsigned int *d_edgeList;
    unsigned int *d_dist;
    PartPointer *d_partNodePointer;

    bool finished = false;
    bool *d_finished;

    bool plus = true;
    bool *d_plus;

    bool *d_sighed;

    gpuErrorcheck(cudaMalloc(&d_nodePointer, num_nodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&d_edgeList, (num_edges + num_nodes) * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&d_dist, num_nodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
    gpuErrorcheck(cudaMalloc(&d_plus, sizeof(bool)));
    gpuErrorcheck(cudaMalloc(&d_partNodePointer, vGraph.numParts * sizeof(PartPointer)));
    gpuErrorcheck(cudaMalloc(&d_sighed,  vGraph.numParts * sizeof(bool)));
    gpuErrorcheck(
            cudaMemcpy(d_nodePointer, vGraph.nodePointer, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_edgeList, vGraph.edgeList, (num_edges + num_nodes) * sizeof(unsigned int),
                             cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_dist, dist, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_partNodePointer, vGraph.partNodePointer, vGraph.numParts * sizeof(PartPointer),
                             cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemset(d_sighed, false, vGraph.numParts * sizeof(bool)));

   

    int itr = 0;
    uint level = 0;

    plus = true;
    Timer t{};
    t.Start();
    while (!finished) {

        itr++;

        if (plus) {
            level++;
        } else {
            plus = true;
        }

        finished = true;

        cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_plus, &plus, sizeof(bool), cudaMemcpyHostToDevice);

        kernel<<< vGraph.numParts / 512 + 1, 512 >>>(
                Part_Size,
                d_sighed,
                vGraph.numParts,
                d_nodePointer,
                d_partNodePointer,
                d_edgeList,
                d_dist,
                d_finished,
                d_plus,
                level);

       cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost);
       cudaMemcpy(&plus, d_plus, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    double runtime = t.Finish();


    cout << "Number of iterations = " << itr << endl;

    cout << "Processing finished in " << runtime << " (ms).\n";

    gpuErrorcheck(cudaMemcpy(dist, d_dist, num_nodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    utility::PrintMaxMin(dist, num_nodes);

    gpuErrorcheck(cudaFree(d_nodePointer));
    gpuErrorcheck(cudaFree(d_edgeList));
    gpuErrorcheck(cudaFree(d_dist));
    gpuErrorcheck(cudaFree(d_finished));
    gpuErrorcheck(cudaFree(d_partNodePointer));
    gpuErrorcheck(cudaFree(d_sighed));

    return 0;

}




