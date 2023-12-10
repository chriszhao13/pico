//
// Created by Chris on 2022/11/15.
//

#include "virtual_graph.h"



//  初始化 节点初始化 从零开始；
// 初始化  度数初始化 有 outdegree 和 indegree

VirtualGraph::VirtualGraph(Graph &graph, uint Part_Size) {

    this->Part_Size = Part_Size;

    if(graph.hasZeroID == false)
	{
		for(int i=0; i< graph.num_edges; i++)
		{
			graph.edges[i].source = graph.edges[i].source - 1;
			graph.edges[i].end = graph.edges[i].end - 1;
		}
	}

    this->graph = &graph;

//  inDegree = new uint [graph.num_nodes];
    outDegree = new uint [graph.num_nodes];

    //度数初始化
    for(int i = 0; i < graph.num_nodes; i++){

        outDegree[i] = 0;
        //inDegree[i] = 0;

    }
    // 无向图
    for (int i = 0; i < graph.num_edges; i++) {

        outDegree[graph.edges[i].source]++;

        outDegree[graph.edges[i].end]++;

    }

}

void VirtualGraph::PrintGraph() {
    cout<<"************************"<<endl;

    int changexx = 0;

    for(int i = 0; i < numParts; i++){
        cout<<RED<<" node: "<<partNodePointer[i].node << RESET;
    }
    cout<<endl;
    for(int i = 0; i < numParts; i++){

        if(changexx != partNodePointer[i].node){
            cout<<endl<<"************************"<<endl;
            cout<<endl;

            for (int j = 0; j < outDegree[changexx]; j ++) {

                cout << " "<<edgeList[nodePointer[changexx] + 1 + j]<<" " ;

            }

            changexx = partNodePointer[i].node;
            cout<<endl<<"-------------------"<<endl;

        }

        cout<<RED<<"node: "<<partNodePointer[i].node <<RESET<<endl;

        cout<<"part: "<<partNodePointer[i].part << endl;
        if(partNodePointer[i].node == 2){cout<<YELLOW<<"WOW!!!!"<<RESET<<endl;}
        unsigned int thisPointer = nodePointer[partNodePointer[i].node];

        int ofs = thisPointer + 1 + partNodePointer[i].part * Part_Size;

        int limit = thisPointer + 1 + outDegree[partNodePointer[i].node];


        for (int j = 0; j < Part_Size && j + ofs < limit; j ++) {

            cout << " "<<edgeList[ofs + j] <<" ";

        }

        cout<<endl;
    }

}

void VirtualGraph::MakeUGraph()
{
    nodePointer = new uint[graph->num_nodes];

    edgeList = new uint[2 * graph->num_edges + graph->num_nodes];

    uint *outDegreeCounter;

    uint source;

    uint end;

    long long counter = 0;

    numParts = 0;

    int numZero = 0;

    // edgelist = outdegree + list
    // numPart 一种有多少个Part; numPart

    for (int i = 0; i < graph->num_nodes ; i++) {

        nodePointer[i] = counter;

        edgeList[counter] = outDegree[i];

        if(outDegree[i] == 0){
            numZero++;
        }

        // numParts

        if(outDegree[i] % Part_Size == 0)
            numParts += outDegree[i] / Part_Size;
        else
            numParts += outDegree[i] / Part_Size + 1;

        // 这里的 +1 是因为 edgelist 存了节点的 度数 信息
        counter = counter + (outDegree[i] + 1);

    }

    // 2. edgelist = degree + nbrList;
    //  初始化为 0

    outDegreeCounter = new uint[graph->num_nodes]();
    // 存完 end 也要存 source
    for (int i = 0; i < graph->num_edges; i++) {

        source = graph->edges[i].source;

        end = graph->edges[i].end;

        // 起始位置的下一个位置 + 1 开始存 边；
        uint location_end = nodePointer[source] + outDegreeCounter[source] + 1;
        uint location_source = nodePointer[end] + outDegreeCounter[end] + 1;
        // 存边

        edgeList[location_end] = end;
        // 偏移自增
        outDegreeCounter[source]++;

        edgeList[location_source] = source;

        outDegreeCounter[end]++;

    }

    // VGrapp 属性

    // partNodePoint： i P
    // 最终得到 虚拟图

    partNodePointer = new PartPointer[numParts];

    int thisNumParts;

    long long countParts = 0;

    for (int i = 0; i < graph->num_nodes; i ++) {

        if(outDegree[i] % Part_Size == 0)
            thisNumParts = outDegree[i] / Part_Size;
        else
            thisNumParts = outDegree[i] / Part_Size + 1;

        // 节点 i 有 j 个part
        for (int j = 0; j < thisNumParts; j ++) {

            partNodePointer[countParts].node = i;

            partNodePointer[countParts].part = j;

            countParts ++;
        }
    }
}