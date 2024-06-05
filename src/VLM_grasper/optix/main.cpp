#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <iostream>
#include <optix_function_table_definition.h> // Include this only in main.cpp
#include "VLM_grasper/optix_wrapper.h"

int main() {
    OptixWrapper optixWrapper;
    optixWrapper.initialize();

    std::cout << "OptiX context initialized in main!" << std::endl;

    return 0;
}
