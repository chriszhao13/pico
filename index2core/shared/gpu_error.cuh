//
// Created by Chris on 2022/12/1.
//

#ifndef LEARN_GRAPH_COMPUTING_GPU_ERROR_CUH
#define LEARN_GRAPH_COMPUTING_GPU_ERROR_CUH

#define gpuErrorcheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


#endif //LEARN_GRAPH_COMPUTING_GPU_ERROR_CUH
