#include "openposeResize.hpp"

const auto THREADS_PER_BLOCK_1D = 16u;
const auto CUDA_NUM_THREADS = 512u;

template<typename T>
inline __device__ T fastMax(const T a, const T b)
{
    return (a > b ? a : b);
}

template<typename T>
inline __device__ T fastMin(const T a, const T b)
{
    return (a < b ? a : b);
}

template<class T>
inline __device__ T fastTruncate(T value, T min = 0, T max = 1)
{
    return fastMin(max, fastMax(min, value));
}


inline unsigned int getNumberCudaBlocks(const unsigned int totalRequired, const unsigned int numberCudaThreads = CUDA_NUM_THREADS)
{
    return (totalRequired + numberCudaThreads - 1) / numberCudaThreads;
}


template <typename T>
inline __device__ void cubicSequentialData(int* xIntArray, int* yIntArray, T& dx, T& dy, const T xSource, const T ySource, const int width, const int height)
{
    xIntArray[1] = fastTruncate(int(xSource + 1e-5), 0, width - 1);
    xIntArray[0] = fastMax(0, xIntArray[1] - 1);
    xIntArray[2] = fastMin(width - 1, xIntArray[1] + 1);
    xIntArray[3] = fastMin(width - 1, xIntArray[2] + 1);
    dx = xSource - xIntArray[1];

    yIntArray[1] = fastTruncate(int(ySource + 1e-5), 0, height - 1);
    yIntArray[0] = fastMax(0, yIntArray[1] - 1);
    yIntArray[2] = fastMin(height - 1, yIntArray[1] + 1);
    yIntArray[3] = fastMin(height - 1, yIntArray[2] + 1);
    dy = ySource - yIntArray[1];
}

template <typename T>
inline __device__ T cubicInterpolation(const T v0, const T v1, const T v2, const T v3, const T dx)
{
    // http://www.paulinternet.nl/?page=bicubic
    // const auto a = (-0.5f * v0 + 1.5f * v1 - 1.5f * v2 + 0.5f * v3);
    // const auto b = (v0 - 2.5f * v1 + 2.0 * v2 - 0.5 * v3);
    // const auto c = (-0.5f * v0 + 0.5f * v2);
    // out = ((a * dx + b) * dx + c) * dx + v1;
    return (-0.5f * v0 + 1.5f * v1 - 1.5f * v2 + 0.5f * v3) * dx * dx * dx
            + (v0 - 2.5f * v1 + 2.f * v2 - 0.5f * v3) * dx * dx
            - 0.5f * (v0 - v2) * dx // + (-0.5f * v0 + 0.5f * v2) * dx
            + v1;
}

template <typename T>
inline __device__ T cubicResize(const T* const sourcePtr, const T xSource, const T ySource, const int widthSource, const int heightSource, const int widthSourcePtr)
{
    int xIntArray[4];
    int yIntArray[4];
    T dx;
    T dy;
    cubicSequentialData(xIntArray, yIntArray, dx, dy, xSource, ySource, widthSource, heightSource);

    T temp[4];
    for (unsigned char i = 0; i < 4; i++)
    {
        const int offset = yIntArray[i]*widthSourcePtr;
        temp[i] = cubicInterpolation(sourcePtr[offset + xIntArray[0]], sourcePtr[offset + xIntArray[1]], sourcePtr[offset + xIntArray[2]], sourcePtr[offset + xIntArray[3]], dx);
    }
    return cubicInterpolation(temp[0], temp[1], temp[2], temp[3], dy);
}

template <typename T>
__global__ void resizeKernel(T* targetPtr, const T* const sourcePtr, const int sourceWidth, const int sourceHeight, const int targetWidth, const int targetHeight)
{
    const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
    const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < targetWidth && y < targetHeight)
    {
        const auto scaleWidth = targetWidth / T(sourceWidth);
        const auto scaleHeight = targetHeight / T(sourceHeight);
        const T xSource = (x + 0.5f) / scaleWidth - 0.5f;
        const T ySource = (y + 0.5f) / scaleHeight - 0.5f;

        targetPtr[y*targetWidth+x] = cubicResize(sourcePtr, xSource, ySource, sourceWidth, sourceHeight, sourceWidth);
    }
}

template <typename T>
void resizeAndMergeGpu(T* targetPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize)
{
    const auto num = sourceSize[0];
    const auto channels = sourceSize[1];
    const auto sourceHeight = sourceSize[2];
    const auto sourceWidth = sourceSize[3];
    const auto targetHeight = targetSize[2];
    const auto targetWidth = targetSize[3];

    const dim3 threadsPerBlock{THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D};
    const dim3 numBlocks{getNumberCudaBlocks(targetWidth, threadsPerBlock.x), getNumberCudaBlocks(targetHeight, threadsPerBlock.y)};
    const auto sourceChannelOffset = sourceHeight * sourceWidth;
    const auto targetChannelOffset = targetWidth * targetHeight;

    for (auto n = 0; n < num; n++)
        for (auto c = 0; c < channels; c++)
            resizeKernel<<<numBlocks, threadsPerBlock>>>(targetPtr + (n*channels + c) * targetChannelOffset, sourcePtr + (n*channels + c) * sourceChannelOffset,
                    sourceWidth, sourceHeight, targetWidth, targetHeight);
}

template void resizeAndMergeGpu(float* targetPtr, const float* const sourcePtr, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);
template void resizeAndMergeGpu(double* targetPtr, const double* const sourcePtr, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);
