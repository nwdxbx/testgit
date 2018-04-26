#include "property.hpp"
const std::map<unsigned int, std::string> property::POSE_COCO_BODY_PARTS {
    {0,  "Nose"},
        {1,  "Neck"},
        {2,  "RShoulder"},
        {3,  "RElbow"},
        {4,  "RWrist"},
        {5,  "LShoulder"},
        {6,  "LElbow"},
        {7,  "LWrist"},
        {8,  "RHip"},
        {9,  "RKnee"},
        {10, "RAnkle"},
        {11, "LHip"},
        {12, "LKnee"},
        {13, "LAnkle"},
        {14, "REye"},
        {15, "LEye"},
        {16, "REar"},
        {17, "LEar"},
        {18, "Background"}
};
const unsigned int property::POSE_COCO_NUMBER_PARTS           = 18u;
const std::vector<unsigned int> property::POSE_COCO_MAP_IDX   {31,32, 39,40, 33,34, 35,36, 41,42, 43,44, 19,20, 21,22, 23,24, 25,26, 27,28, 29,30, 47,48, 49,50, 53,54, 51,52, 55,56, 37,38, 45,46};
const std::vector<unsigned int> property::POSE_COCO_PAIRS     {1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   1,8,   8,9,   9,10, 1,11,  11,12, 12,13,  1,0,   0,14, 14,16,  0,15, 15,17,   2,16,  5,17};
const std::vector<unsigned int> property::POSE_COCO_PAIRS_RENDER POSE_COCO_PAIRS_TO_RENDER;

const std::map<unsigned int, std::string> property::POSE_MPI_BODY_PARTS{
    {0,  "Head"},
        {1,  "Neck"},
        {2,  "RShoulder"},
        {3,  "RElbow"},
        {4,  "RWrist"},
        {5,  "LShoulder"},
        {6,  "LElbow"},
        {7,  "LWrist"},
        {8,  "RHip"},
        {9,  "RKnee"},
        {10, "RAnkle"},
        {11, "LHip"},
        {12, "LKnee"},
        {13, "LAnkle"},
        {14, "Chest"},
        {15, "Background"}
};
const unsigned int property::POSE_MPI_NUMBER_PARTS            = 15u; // Equivalent to size of std::map POSE_MPI_NUMBER_PARTS - 1 (removing background)
const std::vector<unsigned int> property::POSE_MPI_MAP_IDX    {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43};
const std::vector<unsigned int> property::POSE_MPI_PAIRS      POSE_MPI_PAIRS_TO_RENDER;
const std::vector<unsigned int> property::POSE_MPI_PAIRS_RENDER      POSE_MPI_PAIRS_TO_RENDER;

// Constant Global Parameters
const unsigned int property::POSE_MAX_PEOPLE = 96u;

// Constant Array Parameters
const std::array<float, (int)property::PoseModel::Size>               property::POSE_CCN_DECREASE_FACTOR({{   8.f,                    8.f,                    8.f}});
const std::array<unsigned int, (int)property::PoseModel::Size>        property::POSE_MAX_PEAKS({{             POSE_MAX_PEOPLE,        POSE_MAX_PEOPLE,        POSE_MAX_PEOPLE}});
const std::array<unsigned int, (int)property::PoseModel::Size>        property::POSE_NUMBER_BODY_PARTS({{     POSE_COCO_NUMBER_PARTS, POSE_MPI_NUMBER_PARTS,  POSE_MPI_NUMBER_PARTS}});
const std::array<std::vector<unsigned int>, 3>              property::POSE_BODY_PART_PAIRS({{       POSE_COCO_PAIRS,        POSE_MPI_PAIRS,         POSE_MPI_PAIRS}});
const std::array<std::vector<unsigned int>, 3>              property::POSE_MAP_IDX({{               POSE_COCO_MAP_IDX,      POSE_MPI_MAP_IDX,       POSE_MPI_MAP_IDX}});
const std::array<std::string, (int)property::PoseModel::Size> property::POSE_PROTOTXT({{  "pose/coco/pose_deploy_linevec_p",
        "pose/mpi/pose_deploy_linevec_p",
        "pose/mpi/pose_deploy_linevec_faster_4_stages_p"}});
const std::array<std::string, (int)property::PoseModel::Size> property::POSE_TRAINED_MODEL({{ "pose/coco/pose_iter_440000_c",
        "pose/mpi/pose_iter_160000_c",
        "pose/mpi/pose_iter_160000_c"}});

const std::array<float, (int)property::PoseModel::Size>           property::POSE_DEFAULT_NMS_THRESHOLD{{                     0.05f,      0.6f,       0.3f}};
const std::array<unsigned int, (int)property::PoseModel::Size>   property::POSE_DEFAULT_CONNECT_INTER_MIN_ABOVE_THRESHOLD{{ 9,          8,          8}};
const std::array<float, (int)property::PoseModel::Size>           property::POSE_DEFAULT_CONNECT_INTER_THRESHOLD{{           0.05f,      0.01f,      0.01f}};
const std::array<unsigned int, (int)property::PoseModel::Size>   property::POSE_DEFAULT_CONNECT_MIN_SUBSET_CNT{{            3,          3,          3}};
const std::array<float, (int)property::PoseModel::Size>           property::POSE_DEFAULT_CONNECT_MIN_SUBSET_SCORE{{          0.4f,       0.4f,       0.4f}};

const float property::POSE_DEFAULT_ALPHA_POSE = 0.6f;
const float property::POSE_DEFAULT_ALPHA_HEATMAP = 0.7f;

const unsigned int property::FACE_MAX_FACES=96u;
const unsigned int property::FACE_NUMBER_PARTS=70u;
const std::vector<unsigned int> property::FACE_PAIRS_RENDER {FACE_PAIRS_RENDER_GPU};
const float property::FACE_CCN_DECREASE_FACTOR=8.f;
const unsigned int property::FACE_MAX_PEAKS=64;
const std::string property::FACE_PROTOTXT="face/pose_deploy_p";
const std::string property::FACE_TRAINED_MODEL="face/pose_iter_116000_c";
const float property::FACE_DEFAULT_NMS_THRESHOLD=0.1f;
const float property::FACE_DEFAULT_ALPHA_KEYPOINT=0.6f;
const float property::FACE_DEFAULT_ALPHA_HEAT_MAP=0.7f;
