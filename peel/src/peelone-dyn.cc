
__global__ void selectNodesAtLevel1(unsigned int *degrees, unsigned int level, unsigned int V, 
                 unsigned int* bufTails, unsigned int* glBuffers){

    __shared__ unsigned int* glBuffer; 
    __shared__ unsigned int bufTail; 
   
    if(THID == 0){
        bufTail = 0;
       
        glBuffer = glBuffers + blockIdx.x*GLBUFFER_SIZE;
    }
  
    __syncthreads();
   
    unsigned int global_threadIdx = blockIdx.x * blockDim.x + threadIdx.x; 
    for(unsigned int base = 0; base < V; base += N_THREADS){
      
        unsigned int v = base + global_threadIdx; 
      
        if(v >= V) continue;

        if(degrees[v] == level){
            unsigned int loc = atomicAdd(&bufTail, 1);
           
            writeToBuffer(glBuffer, loc, v);
        }
    }
  
    __syncthreads();
   

    
    if(THID == 0) 
    {
        bufTails [blockIdx.x] = bufTail;
    }
}

__global__ void processNodes1(G_pointers d_p, int level, int V, 
                    unsigned int* bufTails, unsigned int* glBuffers, 
                    unsigned int *global_count){
    __shared__ unsigned int bufTail;
    __shared__ unsigned int* glBuffer;
    __shared__ unsigned int base;
  
    unsigned int warp_id = THID / 32;

    unsigned int lane_id = THID % 32;
    unsigned int regTail;
    unsigned int i;
    if(THID==0){
  
        bufTail = bufTails[blockIdx.x];
        base = 0;

        glBuffer = glBuffers + blockIdx.x*GLBUFFER_SIZE; 
        assert(glBuffer!=NULL);
    }

  while(true){
        __syncthreads(); 
        if(base == bufTail) break; 
        i = base + warp_id;
        regTail = bufTail;
        __syncthreads();

        if(i >= regTail) continue; 

        if(THID == 0){
        
            base += WARPS_EACH_BLK;
            if(regTail < base )
                base = regTail;
        }
     
        unsigned int v = readFromBuffer(glBuffer, i);
        unsigned int start = d_p.neighbors_offset[v];
        unsigned int end = d_p.neighbors_offset[v+1];


        while(true){
            __syncwarp();

            if(start >= end) break;

            unsigned int j = start + lane_id;
            start += WARP_SIZE;
            if(j >= end) continue;

            unsigned int u = d_p.neighbors[j];
            if(*(d_p.degrees+u) > level){
                
                // atomicSub+an assignment operation  == atomicSub_>=k
                unsigned int a = atomicSub(d_p.degrees+u, 1);
                if(a <= level){
                     atomicAdd(d_p.degrees+u, 1);
                }



                if(a == level+1){
                    unsigned int loc = atomicAdd(&bufTail, 1);

                        writeToBuffer(glBuffer, loc, u);
                }

               
            }
        }

    }

    if(THID == 0 && bufTail>0){
        atomicAdd(global_count, bufTail); 
    }
}


int peelone_dyn(Graph &data_graph){

    G_pointers data_pointers;

    malloc_graph_gpu_memory(data_graph, data_pointers);

    unsigned int level = 0;
    unsigned int count = 0;
    unsigned int* global_count  = NULL;
    unsigned int* bufTails  = NULL;
    unsigned int* glBuffers     = NULL;

    chkerr(cudaMalloc(&global_count, sizeof(unsigned int)));

    chkerr(cudaMalloc(&bufTails, sizeof(unsigned int)*BLK_NUMS));
    cudaMemset(global_count, 0, sizeof(unsigned int));
    // 0.224GB  glBuffers 
    chkerr(cudaMalloc(&glBuffers,sizeof(unsigned int)*BLK_NUMS*GLBUFFER_SIZE));
       
    
    auto start = chrono::steady_clock::now();
    while(count < data_graph.V){
        cudaMemset(bufTails, 0, sizeof(unsigned int)*BLK_NUMS);
        //For NVIDIA GeForce RTX 3090
        // Maximum number of threads per multiprocessor:  1536 
        // Number of Multiprocessors: 82 
        // Maximum number of threads per block:           1024
        // It can provide 1536 * 82 physical threads.
        // Thus, BLK_NUMS = 123；BLK_DIM = 1024
        selectNodesAtLevel1<<<BLK_NUMS, BLK_DIM>>>(data_pointers.degrees, level, 
                        data_graph.V, bufTails, glBuffers);

        processNodes1<<<BLK_NUMS, BLK_DIM>>>(data_pointers, level, data_graph.V, 
                        bufTails, glBuffers, global_count);

        chkerr(cudaMemcpy(&count, global_count, sizeof(unsigned int), cudaMemcpyDeviceToHost));    
       
        level++;
    }
    data_graph.kmax = level-1;
    auto end = chrono::steady_clock::now();

    cudaFree(glBuffers);
    free_graph_gpu_memory(data_pointers);


    return chrono::duration_cast<chrono::milliseconds>(end - start).count();

}
