//
// Created by 50661 on 2022/11/15.
//

#ifndef LEARN_GRAPH_COMPUTING_VIRTUAL_GRAPH_H
#define LEARN_GRAPH_COMPUTING_VIRTUAL_GRAPH_H

#include "graph.h"
#include "globals.h"

class VirtualGraph {

public:

    //原图 是 COO 以及相关属性
    Graph *graph;

    uint *edgeList;

    uint *nodePointer;

    uint *inDegree;

    uint *outDegree;

    long long numParts;

    uint Part_Size;

    PartPointer *partNodePointer;

    VirtualGraph(Graph &graph, uint Part_Size);

    void PrintGraph();

    void MakeUGraph();



};


#endif //LEARN_GRAPH_COMPUTING_VIRTUAL_GRAPH_H
