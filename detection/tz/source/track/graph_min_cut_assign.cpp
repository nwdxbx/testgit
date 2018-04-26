#include "graph_min_cut_assign.hpp"
#include "3rd/munkres.h"

void GraphAssign::graphMinCut(cv::Mat &SM, std::vector<int> &assign)
{
    if (SM.rows == 0 || SM.cols == 0)
        return;
    assign.clear();
    saebyn::Matrix<float> SMt(SM.rows, SM.cols);
    for (int i=0; i<SM.rows; i++)
    {
        for (int j=0; j<SM.cols; j++)
        {
            float d=SM.at<float>(i, j);
            SMt(i, j)=d;
        }
    }

    saebyn::Munkres<float> dt;
    dt.solve(SMt);
    //std::cout << SM << std::endl;
    //std::cout << SMt << std::endl;
    for (int i=0; i<SM.rows; i++)
    {
        int as=-1;
        for (int j=0; j<SM.cols; j++)
        {
            if (SMt(i, j)==0)
                as=j;
        }
        assign.push_back(as);
    }
}
