#ifndef ALG_GRAPH_MIN_CUT_ASSIGN_A
#define ALG_GRAPH_MIN_CUT_ASSIGN_A
#include <vector>
#include <opencv2/opencv.hpp>

class GraphAssign
{
public:
    static void graphMinCut(cv::Mat &SM, std::vector<int> &assign);
};

#endif
