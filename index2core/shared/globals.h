//
// Created by Chris on 2022/11/15.
//

#ifndef LEARN_GRAPH_COMPUTING_GLOBALS_H
#define LEARN_GRAPH_COMPUTING_GLOBALS_H
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[38m"      /* White */
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <ctime>
#include <random>
#include <cstdio>
#include <iomanip>
#include <locale>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <string>
#include <cstring>

typedef unsigned int uint;

using namespace std;

// max int
const unsigned int  DIST_INFINITY = std::numeric_limits<unsigned int>::max() - 1;

struct Edge{

    uint source;
    uint end;

};

struct PartPointer{
    uint node;
    uint part;
};

#endif //LEARN_GRAPH_COMPUTING_GLOBALS_H
