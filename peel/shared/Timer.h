//
// Created by Chris on 2022/12/1.
//

#ifndef LEARN_GRAPH_COMPUTING_TIMER_H
#define LEARN_GRAPH_COMPUTING_TIMER_H

#include "globals.h"
#include <stdlib.h>
#include <sys/time.h>

class Timer {

    timeval StartingTime;

public:
    double Finish();
    void Start();
};


#endif //LEARN_GRAPH_COMPUTING_TIMER_H
