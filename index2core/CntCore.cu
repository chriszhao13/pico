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

#define printMinMaxOneStep 1

__device__ int DEVICE_count_change = 0;
__device__ int DEVICE_nbr = 0;

__global__ void core_kernel(

    uint partSize,

    unsigned int numParts,

    unsigned int *nodePointer,

    PartPointer *partNodePointer,

    const unsigned int *edgeList,

    unsigned int *histo,

    unsigned int *core,

    unsigned int *cnt)
{

    unsigned int partId = blockDim.x * blockIdx.x + threadIdx.x;

    if (partId < numParts)
    {

        unsigned int id = partNodePointer[partId].node;

        unsigned int part = partNodePointer[partId].part;

        if (cnt[id] >= core[id])
            return;

        unsigned int thisPointer = nodePointer[id];

        unsigned int degree = edgeList[thisPointer];

        int numParts;

        if (degree % partSize == 0)

            numParts = degree / partSize;

        else

            numParts = degree / partSize + 1;

        unsigned int nbr;

        unsigned int ofs = thisPointer + part + 1;
        //

        for (int i = 0; i < partSize; i++)
        {

            if (part + i * numParts >= degree)
                break;

            nbr = ofs + i * numParts;

            unsigned int ofs_nbr = min(degree, core[edgeList[nbr]]);

            atomicAdd(histo + thisPointer + ofs_nbr, 1);
        }
    }
}

__global__ void count_kernel(

    unsigned int num_node,

    unsigned int *nodePointer,

    unsigned int *histo,

    unsigned int *core,

    const unsigned int *edgeList,

    unsigned int *cnt,

    bool *finished,

    bool *sighed)
{

    unsigned int nodeId = blockDim.x * blockIdx.x + threadIdx.x;

    if (cnt[nodeId] >= core[nodeId] || nodeId >= num_node)
        return;

    unsigned int thisPointer = nodePointer[nodeId];

    unsigned int node_old_core = core[nodeId];

    unsigned int sum = 0;

    unsigned int k;

    for (k = edgeList[thisPointer]; k >= 1; k--)
    {

        sum += histo[thisPointer + k];

        if (sum >= k)
            break;
    }
    if (k == core[nodeId])
        return;

    *finished = false;
    histo[thisPointer] = node_old_core;
    core[nodeId] = k;

    sighed[nodeId] = true;
}

__global__ void nbr_kernel(

    uint partSize,

    unsigned int numParts,

    unsigned int *nodePointer,

    PartPointer *partNodePointer,

    unsigned int *edgeList,

    bool *sighed,

    bool *sighedNBR,

    unsigned int *cnt)
{

    unsigned int partId = blockDim.x * blockIdx.x + threadIdx.x;

    if (partId < numParts)
    {

        unsigned int id = partNodePointer[partId].node;

        unsigned int part = partNodePointer[partId].part;
        // CSR

        if (!sighed[id])
        {
            return;
        }

        unsigned int thisPointer = nodePointer[id];
        // 前 n 个
        unsigned int degree = edgeList[thisPointer];

        int numParts;

        if (degree % partSize == 0)

            numParts = degree / partSize;

        else

            numParts = degree / partSize + 1;

        unsigned int nbr;

        unsigned int ofs = thisPointer + part + 1;

        for (int i = 0; i < partSize; i++)
        {

            if (part + i * numParts >= degree)
                break;

            nbr = ofs + i * numParts;
            cnt[edgeList[nbr]] = 0;
            sighedNBR[edgeList[nbr]] = true;
        }
    }
}

__global__ void cnt_kernel(

    uint partSize,

    unsigned int numParts,

    unsigned int *nodePointer,

    PartPointer *partNodePointer,

    const unsigned int *edgeList,

    unsigned int *histo,

    unsigned int *core,

    unsigned int *cnt,

    bool *sighedNBR)
{

    unsigned int partId = blockDim.x * blockIdx.x + threadIdx.x;

    if (partId < numParts)
    {

        unsigned int id = partNodePointer[partId].node;

        unsigned int part = partNodePointer[partId].part;

        if (!sighedNBR[id])
        {
            return;
        }

        unsigned int thisPointer = nodePointer[id];

        unsigned int degree = edgeList[thisPointer];

        int numParts;

        if (degree % partSize == 0)

            numParts = degree / partSize;

        else

            numParts = degree / partSize + 1;

        unsigned int nbr;

        unsigned int ofs = thisPointer + part + 1;

        for (int i = 0; i < partSize; i++)
        {

            if (part + i * numParts >= degree)
                break;

            nbr = ofs + i * numParts;

            if (core[edgeList[nbr]] >= core[id])
            {
                atomicAdd(cnt + id, 1);
            }
        }
    }
}

int main(int argc, char **argv)
{

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

    uint *core = new uint[num_nodes];

    for (int i = 0; i < num_nodes; i++)
    {
        core[i] = vGraph.outDegree[i];
    }
   
    unsigned int *d_nodePointer;
    unsigned int *d_edgeList;
    unsigned int *d_core;
    unsigned int *d_degree;
    unsigned int *d_histo;
    unsigned int *d_cnt;
    PartPointer *d_partNodePointer;

    bool finished = true;
    bool *d_finished;

    bool *d_sighed;

    bool *d_sighedNBR;

    gpuErrorcheck(cudaMalloc(&d_nodePointer, num_nodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&d_edgeList, (num_edges + num_nodes) * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&d_core, num_nodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&d_cnt, num_nodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&d_degree, num_nodes * sizeof(unsigned int)));

    gpuErrorcheck(cudaMalloc(&d_histo, (num_edges + num_nodes) * sizeof(unsigned int)));

    gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));

    gpuErrorcheck(cudaMalloc(&d_partNodePointer, vGraph.numParts * sizeof(PartPointer)));
    gpuErrorcheck(cudaMalloc(&d_sighed, num_nodes * sizeof(bool)));
    gpuErrorcheck(cudaMalloc(&d_sighedNBR, num_nodes * sizeof(bool)));
    gpuErrorcheck(
        cudaMemcpy(d_nodePointer, vGraph.nodePointer, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_edgeList, vGraph.edgeList, (num_edges + num_nodes) * sizeof(unsigned int),
                             cudaMemcpyHostToDevice));

    gpuErrorcheck(cudaMemcpy(d_core, core, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));

    gpuErrorcheck(cudaMemcpy(d_degree, core, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));

    gpuErrorcheck(cudaMemcpy(d_partNodePointer, vGraph.partNodePointer, vGraph.numParts * sizeof(PartPointer),
                             cudaMemcpyHostToDevice));

    gpuErrorcheck(cudaMemset(d_cnt, 0, num_nodes * sizeof(unsigned int)));

   
    int itr = 0;
    Timer t{};
    t.Start();

    while (true)
    {

        itr++;
        finished = true;
        cudaMemset(d_sighed, false, num_nodes * sizeof(bool));
        cudaMemset(d_sighedNBR, false, num_nodes * sizeof(bool));
        cudaMemset(d_histo, 0, (num_edges + num_nodes) * sizeof(unsigned int));
        cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice);

        core_kernel<<<vGraph.numParts / 512 + 1, 512>>>(
            Part_Size,

            vGraph.numParts,

            d_nodePointer,

            d_partNodePointer,

            d_edgeList,

            d_histo,

            d_core,

            d_cnt);

        count_kernel<<<num_nodes / 512 + 1, 512>>>(
            num_nodes,
            d_nodePointer,
            d_histo,
            d_core,
            d_edgeList,
            d_cnt,
            d_finished,
            d_sighed);

        cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost);

        if (finished)
            break;

        nbr_kernel<<<vGraph.numParts / 512 + 1, 512>>>(
            Part_Size,

            vGraph.numParts,

            d_nodePointer,

            d_partNodePointer,
            // 边数组
            d_edgeList,

            d_sighed,

            d_sighedNBR,

            d_cnt);

        cnt_kernel<<<vGraph.numParts / 512 + 1, 512>>>(
            Part_Size,

            vGraph.numParts,

            d_nodePointer,

            d_partNodePointer,
            d_edgeList,
            // 边数组
            d_histo,
            d_core,
            d_cnt,
            d_sighedNBR);
    }
    //
    double runtime = t.Finish();
    //
    cout << "Number of iterations = " << itr << endl;

    cout << "Processing finished in " << runtime << " (ms).\n";

    gpuErrorcheck(cudaMemcpy(core, d_core, num_nodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    utility::PrintMaxMin(core, num_nodes);

    gpuErrorcheck(cudaFree(d_nodePointer));
    gpuErrorcheck(cudaFree(d_edgeList));
    gpuErrorcheck(cudaFree(d_core));
    gpuErrorcheck(cudaFree(d_finished));
    gpuErrorcheck(cudaFree(d_partNodePointer));
    gpuErrorcheck(cudaFree(d_sighed));
    gpuErrorcheck(cudaFree(d_cnt));
    gpuErrorcheck(cudaFree(d_histo))
    gpuErrorcheck(cudaFree(d_degree))

            return 0;
}
