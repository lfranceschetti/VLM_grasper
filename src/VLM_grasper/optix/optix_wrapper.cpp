#include "VLM_grasper/optix_wrapper.h"
#include <optix_stubs.h> // Include this to resolve the undefined references


OptixWrapper::OptixWrapper() : context(nullptr), cuCtx(nullptr) {}

OptixWrapper::~OptixWrapper() {
    if (context) {
        optixDeviceContextDestroy(context);
    }
    if (cuCtx) {
        cuCtxDestroy(cuCtx);
    }
}

void OptixWrapper::initialize() {
    // Initialize CUDA
    cudaFree(0);

    // Initialize the OptiX API
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = nullptr;
    options.logCallbackData = nullptr;
    options.logCallbackLevel = 4;

    CUresult cuRes = cuCtxCreate(&cuCtx, 0, 0);
    if (cuRes != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorName(cuRes, &errStr);
        std::cerr << "Error creating CUDA context: " << errStr << std::endl;
        return;
    }

    OptixResult optixRes = optixDeviceContextCreate(cuCtx, 0, &context);
    if (optixRes != OPTIX_SUCCESS) {
        std::cerr << "Error creating OptiX context: " << optixGetErrorName(optixRes) << std::endl;
        return;
    }

    std::cout << "OptiX context successfully created in wrapper!" << std::endl;
}
