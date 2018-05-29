#define CPU_ONLY  0

#include <iostream>
#include "zhihui_alg_interface.hpp"
#include <opencv2/opencv.hpp>


using namespace cv;  
using namespace std;  
  
int main(int argc, char ** argv)
{
    Mat img1=imread("./1.jpg");
    Mat img2=imread("./2.jpg");

//    Mat img1=imread("/media/e/mx_face/datasets/megaface_data/renlian/46976970@N00_identity_8/2331657915_0.jpg");
//    Mat img2=imread("/media/e/mx_face/datasets/megaface_data/renlian/46976970@N00_identity_8/257385685_0.jpg");

//    Mat img1=imread("/media/e/zhihui/picture/p1/3620614060B811E8AA332CFDA1C74FA5.png");
//    Mat img2=imread("/media/e/zhihui/picture/p2/941ebe00d02b460ea19759343cedd942.PNG");
    if(!img1.data) {cout<<"error1";return -1;}   //检查有没有读到图片  
    if(!img2.data) {cout<<"error2";return -1;}   //检查有没有读到图片

    //namedWindow("image1");//有两个参数，第二个有默认形参
    //imshow("image1",img1);
    //namedWindow("image2");//有两个参数，第二个有默认形参
    //imshow("image2",img2);
    waitKey(100);    

    ZhihuiPara stPara;
    if(argc > 1)
        stPara.deviceID = atoi(argv[1]);
    else
        stPara.deviceID = 0;

    stPara.structsize = sizeof(ZhihuiPara);


    void *pZhihuiHandle;
    string result1, result2;
    float fsimilar = -1.0f;

    zhihui_open(&pZhihuiHandle, &stPara);
    double t1 = (double)getTickCount();
    zhihui_facecal_process(pZhihuiHandle, img1, result1);
    zhihui_facecal_process(pZhihuiHandle, img2, result2);
    double t2 = (double)getTickCount();

    if(result1.empty() || result2.empty())
    {
        cout << "no face detect" << endl;
        return 0;
    }

    //compare 2000 times, test time
    for(int i=0; i<2000; i++)
    {
        zhihui_facecompare_process(result1, result2, fsimilar);
    }



    double t3 = (double)getTickCount();

    cout<<"extract feature time:"<<(t2-t1)*1000/getTickFrequency()<<endl;
    cout<<"compare time:"<<(t3-t2)*1000/getTickFrequency()<<endl;
    zhihui_close(pZhihuiHandle);

    cout << "similarity = " << fsimilar << endl;


    string vision;
    zhihui_getvision(vision);
    cout << "current vision: " << vision <<endl;


    waitKey(1000);
    return 0; 
}
