#ifndef VLM_GRASPER_OPTIX_WRAPPER_H
#define VLM_GRASPER_OPTIX_WRAPPER_H

#include <optix.h>
#include <cuda_runtime.h>
#include <iostream>

class OptixWrapper {
public:
    OptixWrapper();
    ~OptixWrapper();

    void initialize();

private:
    OptixDeviceContext context;
    CUcontext cuCtx;
};

#endif // VLM_GRASPER_OPTIX_WRAPPER_H
