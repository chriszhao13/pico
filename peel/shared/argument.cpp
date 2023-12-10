//
// Created by Chris on 2022/11/30.
//

#include "argument.h"

argument::argument(int argc, char **argv, bool canHaveSource, bool canHaveIters) {

    this->argc = argc;

    this->argv = argv;

    this->canHaveSource = canHaveSource;
    hasSourceNode = false;
    sourceNode = 0;

    this->canHaveIters = canHaveIters;
    hasNumberOfIters = false;
    numberOfIters = 1;

    sourceNode = 0;

    deviceID = 0;


    hasInput = false;

    hasOutput = false;

    hasDeviceID = false;

    Parse();

}

bool argument::Parse() {

    try{

        if(argc != 2){
            cout<<GenerateHelpString();
            exit(0);
        }
        input = string (argv[1]);
        hasInput = true;

    }
    catch(const std::exception &strExeception){
        std::cerr<<"An exception has occurred.\n";
        GenerateHelpString();
        exit(0);
    }
    catch(...){
        std::cerr<<"An exception has occurred.\n";
        GenerateHelpString();
        exit(0);
    }

    if(hasInput)
        return true;
    else{
        cout<< "\n input graph file argument is required. \n";
        cout<< GenerateHelpString();
        exit(0);
    }
}
string argument::GenerateHelpString() {

    string str = "\nRequired arguments:";
    str += "\n    [--input]: Input graph file. E.g., --input FacebookGraph.txt";
    str += "\nOptional arguments";
    if(canHaveSource)
        str += "\n    [--source]:  Begins from the source (Default: 0). E.g., --source 10";
    str += "\n    [--output]: Output file for results. E.g., --output results.txt";
    str += "\n    [--device]: Select GPU device (default: 0). E.g., --device 1";
    if(canHaveIters)
        str += "\n    [--iteration]: Number of iterations (default: 1). E.g., --iterations 10";
    str += "\n\n";
    return str;

}

