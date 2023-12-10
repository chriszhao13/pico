//
// Created by Chris on 2022/11/14.
//

#ifndef LEARN_GRAPH_COMPUTING_GRAPH_H
#define LEARN_GRAPH_COMPUTING_GRAPH_H

#include "globals.h"

// Graph is COO
class Graph{
public:

    string graphFilePath;

    bool isWeighted;

    bool hasZeroID;

    uint num_nodes;

    uint num_edges;

    vector<Edge> edges;

    vector<uint> weights;

    bool graphLoaded;

    Graph(string graphFilePath, bool isWeighted);

    void ReadGraph();

};


#endif //LEARN_GRAPH_COMPUTING_GRAPH_H
