//
// Created by Chris on 2022/12/1.
//

#ifndef LEARN_GRAPH_COMPUTING_UTIL_H
#define LEARN_GRAPH_COMPUTING_UTIL_H

#include "globals.h"

class utility {

public:

    static void PrintMaxMin(uint *result, uint n);
    static unsigned int ReturnMaxMin(uint *result, uint n);
// printf
    static void PrintResults(uint *results, uint n);
    static void PrintResults(float *results, uint n);

    // save
    static void SaveResults(string filepath, uint *results, uint n);
    static void SaveResults(string filepath, float *results, uint n);
};


#endif //LEARN_GRAPH_COMPUTING_UTIL_H
