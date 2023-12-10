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

__global__ void scan_nodes(unsigned int num_node,

                           bool *scatter_flag,

                           bool *delete_flag,

                           unsigned int *core,

                           unsigned int *final_core,

                           unsigned int level,

                           bool *finished,

                           bool *plus
){

    unsigned int Id = blockDim.x * blockIdx.x + threadIdx.x;

    if(delete_flag[Id] || Id >= num_node) return;

    *finished = false;

    if(core[Id] <= level)
    {
        delete_flag[Id] = true;
        scatter_flag[Id] = true;
        final_core[Id] = level;
        *plus = false;
    }

}

__global__ void scatter_edges(
        uint partSize,

        bool *scatter_flag,

        bool *delete_flag,

        unsigned int numParts,

        unsigned int *nodePointer,

        PartPointer *partNodePointer,

        unsigned int *edgeList,

        unsigned int *core,

        unsigned int level) {

    unsigned int partId = blockDim.x * blockIdx.x + threadIdx.x;

    if (partId < numParts) {

        unsigned int id = partNodePointer[partId].node;
        unsigned int part = partNodePointer[partId].part;

        if (!scatter_flag[id]) {
            return;
        }
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
            if(!delete_flag[edgeList[end]]){
                atomicSub(core + edgeList[end], 1);
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

    uint *core = new uint[num_nodes];

    for (int i = 0; i < num_nodes; i++) {
        core[i] = vGraph.outDegree[i];
    }

    unsigned int *d_nodePointer;
    unsigned int *d_edgeList;
    unsigned int *d_core;
    unsigned int *d_final_core;
    PartPointer *d_partNodePointer;

    bool finished = false;
    bool *d_finished;

    bool plus = true;
    bool *d_plus;


    bool *d_scatter_flag;
    bool *d_delete_flag;

    gpuErrorcheck(cudaMalloc(&d_nodePointer, num_nodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&d_edgeList, (num_edges + num_nodes) * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&d_core, num_nodes * sizeof(unsigned int)));
    gpuErrorcheck(cudaMalloc(&d_final_core, num_nodes * sizeof(unsigned int)));

    gpuErrorcheck(cudaMalloc(&d_finished, sizeof(bool)));
    gpuErrorcheck(cudaMalloc(&d_plus, sizeof(bool)));
    gpuErrorcheck(cudaMalloc(&d_partNodePointer, vGraph.numParts * sizeof(PartPointer)));

    gpuErrorcheck(cudaMalloc(&d_scatter_flag,  num_nodes * sizeof(bool)));
    gpuErrorcheck(cudaMalloc(&d_delete_flag,  num_nodes * sizeof(bool)));

    gpuErrorcheck(
            cudaMemcpy(d_nodePointer, vGraph.nodePointer, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_edgeList, vGraph.edgeList, (num_edges + num_nodes) * sizeof(unsigned int),
                             cudaMemcpyHostToDevice));
    gpuErrorcheck(cudaMemcpy(d_core, core, num_nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));


    gpuErrorcheck(cudaMemcpy(d_partNodePointer, vGraph.partNodePointer, vGraph.numParts * sizeof(PartPointer),
                             cudaMemcpyHostToDevice));

    gpuErrorcheck(cudaMemset(d_delete_flag, false, num_nodes * sizeof(bool)));

    Timer t{};
    t.Start();

    int itr = 0;
    uint level = 0;

    plus = true;

    while (!finished) {

        itr++;
        //printf("iter: %d %d", itr, level);
        if (plus) {
            level++;
        } else {
            plus = true;
        }

        finished = true;
        cudaMemset(d_scatter_flag, false, num_nodes * sizeof(bool));
        cudaMemcpy(d_finished, &finished, sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemcpy(d_plus, &plus, sizeof(bool), cudaMemcpyHostToDevice);

        scan_nodes<<<num_nodes / 512 + 1, 512>>>(
                num_nodes,
                d_scatter_flag,
                d_delete_flag,
                d_core,
                d_final_core,
                level,
                d_finished,
                d_plus);
      

        scatter_edges<<< vGraph.numParts / 512 + 1, 512 >>>(
                Part_Size,
                d_scatter_flag,
                d_delete_flag,
                vGraph.numParts,
                d_nodePointer,
                d_partNodePointer,
                d_edgeList,
                d_core,
                level);



        cudaMemcpy(&finished, d_finished, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaMemcpy(&plus, d_plus, sizeof(bool), cudaMemcpyDeviceToHost);

    }

    double runtime = t.Finish();
    cout << "Number of iterations = " << itr << endl;

    // int count_sub = 0;

    // cout <<RED<<"SUB: "<< count_sub <<RESET<<endl;

    cout << "Processing finished in " << runtime << " (ms).\n";

    gpuErrorcheck(cudaMemcpy(core, d_final_core, num_nodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    utility::PrintMaxMin(core, num_nodes);
    // utility::PrintResults(core, 34);
    // if (arguments.hasOutput)
    //     utility::SaveResults(arguments.output, core, num_nodes);

    gpuErrorcheck(cudaFree(d_nodePointer));
    gpuErrorcheck(cudaFree(d_edgeList));
    gpuErrorcheck(cudaFree(d_core));
    gpuErrorcheck(cudaFree(d_final_core));
    gpuErrorcheck(cudaFree(d_finished));
    gpuErrorcheck(cudaFree(d_partNodePointer));
    gpuErrorcheck(cudaFree(d_delete_flag));
    gpuErrorcheck(cudaFree(d_scatter_flag));

    return 0;
}




