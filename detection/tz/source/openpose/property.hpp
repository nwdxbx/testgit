#ifndef ALG_OPENPOSE_POSE_MX_A
#define ALG_OPENPOSE_POSE_MX_A

#include <array>
#include <map>
#include <vector>
#include <string>

class property
{
public:
    enum class PoseModel : unsigned char
    {
        COCO_18 = 0,    /**< COCO model, with 18+1 components (see poseParameters.hpp for details). */
        MPI_15,         /**< MPI model, with 15+1 components (see poseParameters.hpp for details). */
        MPI_15_4,       /**< Same MPI model, but reducing the number of CNN stages to 4 (see poseModel.cpp for details). It should increase speed and reduce accuracy.*/
        Size,
    };

    enum class PoseProperty : unsigned char
    {
        NMSThreshold = 0,
        ConnectInterMinAboveThreshold,
        ConnectInterThreshold,
        ConnectMinSubsetCnt,
        ConnectMinSubsetScore,
        Size,
    };

    enum class ScaleMode : unsigned char
    {
        InputResolution,
        NetOutputResolution,
        OutputResolution,
        ZeroToOne, // [0, 1]
        PlusMinusOne, // [-1, 1]
        UnsignedChar, // [0, 255]
    };

    enum class HeatMapType : unsigned char
    {
        Parts,
        Background,
        PAFs,
    };

    static const std::map<unsigned int, std::string> POSE_COCO_BODY_PARTS;
    static const unsigned int POSE_COCO_NUMBER_PARTS;
    static const std::vector<unsigned int> POSE_COCO_MAP_IDX;
#define POSE_COCO_PAIRS_TO_RENDER                   {1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10, 1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17}
    static const std::vector<unsigned int> POSE_COCO_PAIRS;
    static const std::vector<unsigned int> POSE_COCO_PAIRS_RENDER;

    static const std::map<unsigned int, std::string> POSE_MPI_BODY_PARTS;

    static const unsigned int POSE_MPI_NUMBER_PARTS;
    static const std::vector<unsigned int> POSE_MPI_MAP_IDX;
#define POSE_MPI_PAIRS_TO_RENDER                    { 0,1,    1,2,    2,3,     3,4,   1,5,    5,6,    6,7,    1,14,  14,8,    8,9,   9,10,  14,11,   11,12,  12,13}
    static const std::vector<unsigned int> POSE_MPI_PAIRS;
    static const std::vector<unsigned int> POSE_MPI_PAIRS_RENDER;

    // Constant Global Parameters
    static const unsigned int POSE_MAX_PEOPLE;

    // Constant Array Parameters
    static const std::array<float, (int)PoseModel::Size>               POSE_CCN_DECREASE_FACTOR;
    static const std::array<unsigned int, (int)PoseModel::Size>        POSE_MAX_PEAKS;
    static const std::array<unsigned int, (int)PoseModel::Size>        POSE_NUMBER_BODY_PARTS;
    static const std::array<std::vector<unsigned int>, 3>              POSE_BODY_PART_PAIRS;
    static const std::array<std::vector<unsigned int>, 3>              POSE_MAP_IDX;
    static const std::array<std::string, (int)PoseModel::Size> POSE_PROTOTXT;
    static const std::array<std::string, (int)PoseModel::Size> POSE_TRAINED_MODEL;
    static const std::map<unsigned int, std::string>& getPoseBodyPartMapping(const PoseModel poseModel);

    static const std::array<float, (int)PoseModel::Size>           POSE_DEFAULT_NMS_THRESHOLD;
    static const std::array<unsigned int, (int)PoseModel::Size>   POSE_DEFAULT_CONNECT_INTER_MIN_ABOVE_THRESHOLD;
    static const std::array<float, (int)PoseModel::Size>           POSE_DEFAULT_CONNECT_INTER_THRESHOLD;
    static const std::array<unsigned int, (int)PoseModel::Size>   POSE_DEFAULT_CONNECT_MIN_SUBSET_CNT;
    static const std::array<float, (int)PoseModel::Size>           POSE_DEFAULT_CONNECT_MIN_SUBSET_SCORE;

    static const float POSE_DEFAULT_ALPHA_POSE ;
    static const float POSE_DEFAULT_ALPHA_HEATMAP ;


    
    
    
    
    
    
    static const unsigned int FACE_MAX_FACES;

    static const unsigned int FACE_NUMBER_PARTS;
    #define FACE_PAIRS_RENDER_GPU {0,1,  1,2,  2,3,  3,4,  4,5,  5,6,  6,7,  7,8,  8,9,  9,10,  10,11,  11,12,  12,13,  13,14,  14,15,  15,16,  17,18,  18,19,  19,20, \
                                  20,21,  22,23,  23,24,  24,25,  25,26,  27,28,  28,29,  29,30,  31,32,  32,33,  33,34,  34,35,  36,37,  37,38,  38,39,  39,40,  40,41, \
                                  41,36,  42,43,  43,44,  44,45,  45,46,  46,47,  47,42,  48,49,  49,50,  50,51,  51,52,  52,53,  53,54,  54,55,  55,56,  56,57,  57,58, \
                                  58,59,  59,48,  60,61,  61,62,  62,63,  63,64,  64,65,  65,66,  66,67,  67,60}
    static const std::vector<unsigned int> FACE_PAIRS_RENDER;
    #define FACE_COLORS_RENDER      255.f,    255.f,    255.f

    // Constant parameters
    static const float FACE_CCN_DECREASE_FACTOR;
    static const unsigned int FACE_MAX_PEAKS;
    static const std::string FACE_PROTOTXT;
    static const std::string FACE_TRAINED_MODEL;

    // Default Model Parameters
    // They might be modified on running time
    static const float FACE_DEFAULT_NMS_THRESHOLD;

    // Rendering default parameters
    static const float FACE_DEFAULT_ALPHA_KEYPOINT;
    static const float FACE_DEFAULT_ALPHA_HEAT_MAP;
};
#endif
