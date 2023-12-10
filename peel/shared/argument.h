//
// Created by Chris on 2022/11/30.
//

#ifndef LEARN_GRAPH_COMPUTING_ARGUMENT_H
#define LEARN_GRAPH_COMPUTING_ARGUMENT_H

#include "globals.h"

class argument {

public:
    int argc;
    char** argv;

    bool canHaveSource;
    bool hasSourceNode;
    int sourceNode;

    bool canHaveIters;
    bool hasNumberOfIters;
    int numberOfIters;

    uint Part_Size = 64;


    bool hasInput;
    string input;

    bool hasOutput;
    string output;

    bool hasDeviceID;
    int deviceID = 0;

    argument(int argc, char **argv, bool canHaveSource, bool canHaveIters);

    bool Parse();

    string GenerateHelpString();
};



#endif //LEARN_GRAPH_COMPUTING_ARGUMENT_H
