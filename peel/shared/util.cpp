//
// Created by Chris on 2022/12/1.
//
#include "util.h"

void utility::PrintMaxMin(uint *result, uint n) {

    uint max = 0;
    uint min = 10;
    for (int i = 0; i < n; i++){

        if(max < result[i])
            max = result[i];
        if(min > result[i])
            min = result[i];

    }

    cout << " Max K = " << max <<endl;
    cout << " Min K = " << min <<endl;
}


unsigned int utility::ReturnMaxMin(uint *result, uint n) {

    uint max = 0;
    uint min = 10;

    for (int i = 0; i < n; i++){

        if(max < result[i])
            max = result[i];
        if(min > result[i])
            min = result[i];

    }

    cout << " Max K = " << max <<endl;
    cout << " Min K = " << min <<endl;

    return max;
}

void utility::PrintResults(uint *results, uint n) {

    cout << "Result of first " << n << " nodes:\n[";

    for (int i = 0; i < n; i++) {
        if (i > 0) {
            cout << " ";
        }
        cout << i << ":" << results[i];
    }
    cout << "]\n";
}

void utility::PrintResults(float *results, uint n) {

    cout << "Result of first " << n << " nodes:\n[";

    for (int i = 0; i < n; i++) {
        if (i > 0) {
            cout << " ";
        }
        cout << i << ":" << results[i];
    }
    cout << "]\n";
}

void utility::SaveResults(std::string filepath, uint *results, uint n) {

    cout << "Saving the results into the following file:\n";

    cout << ">> " << filepath << endl;

    ofstream outfile;

    outfile.open(filepath);

    for (int i = 0; i < n; i++) {

        outfile << i << " " << results[i] << endl;

    }

    outfile.close();

    cout << "Done saving.\n";
}

void utility::SaveResults(std::string filepath, float *results, uint n) {

    cout << "Saving the results into " << filepath << " ...... " << flush;
    ofstream outfile;
    outfile.open(filepath);
    for (int i = 0; i < n; i++)
        outfile << i << " " << results[i] << endl;
    outfile.close();
    cout << " Completed.\n";
}