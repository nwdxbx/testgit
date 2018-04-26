#ifndef ALG_OPENPOSE_MX_RESIZE_A
#define ALG_OPENPOSE_MX_RESIZE_A
#include <array>

template <typename T>
void resizeAndMergeGpu(T* targetPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);

#endif

