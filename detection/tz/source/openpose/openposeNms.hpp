#ifndef ALG_OPENPOSE_MX_NMS_A
#define ALG_OPENPOSE_MX_NMS_A
#include <array>

template <typename T>
void nmsGpu(T* targetPtr, int* kernelPtr, const T* const sourcePtr, const T threshold, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);
#endif
