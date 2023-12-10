//
// Created by Chris on 2022/11/15.
//
#include "graph.h"

#include <utility>

Graph::Graph(std::string graphFilePath, bool isWeighted) {

    this->graphFilePath = std::move(graphFilePath);
    this->isWeighted = false;
    graphLoaded = false;
    hasZeroID = false;

}


void Graph::ReadGraph() {

    cout << "Reading the input graph from the following file:\n>> " << graphFilePath << endl;

    ifstream infile;

    infile.open(graphFilePath);

    stringstream ss;

    if (graphLoaded) {
        edges.clear();
        weights.clear();
    }

    uint source;
    uint end;
    uint w8;

    uint i = 0;

    string line;

    unsigned long edgeCounter = 0;

    Edge newEdge{};

    uint max = 0;
    getline(infile, line);
    getline(infile, line);
    while (getline(infile, line)) {

        stringstream node_String_To_Int(line);

        node_String_To_Int >> newEdge.source;
        node_String_To_Int >> newEdge.end;
        if(newEdge.source == newEdge.end) continue;

        // cout<< newEdge.source <<" "<<newEdge.end<<endl;

        edges.push_back(newEdge);

        if (newEdge.source == 0)
            hasZeroID = true;
        if (newEdge.end == 0)
            hasZeroID = true;
        if (max < newEdge.source)
            max = newEdge.source;
        if (max < newEdge.end)
            max = newEdge.end;

        if (isWeighted) {
            if (ss >> w8)
                weights.push_back(w8);
            else
                weights.push_back(1);
        }

        edgeCounter++;

    }

    infile.close();

    graphLoaded = true;

    num_edges = edgeCounter;

    num_nodes = max;

    if (hasZeroID)
        num_nodes++;

    cout << "Done reading.\n";

    cout << "Number of nodes = " << num_nodes << endl;

    cout << "Number of edges = " << num_edges << endl;
}