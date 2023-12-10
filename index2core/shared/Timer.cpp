//
// Created by Chris on 2022/12/1.
//

#include "Timer.h"

void Timer::Start() {

    gettimeofday(&StartingTime, NULL);

}

double Timer::Finish() {

    timeval PausingTime, ElapsedTime;

    gettimeofday(&PausingTime, NULL);
    timersub(&PausingTime, &StartingTime, &ElapsedTime);

    // output time is ms;

    double d = ElapsedTime.tv_sec * 1000.0 + ElapsedTime.tv_usec / 1000.0;

    return d;


}