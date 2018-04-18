#include "../include/tuzhenfeature.h"
#include <string>
#include <iostream>
#include <dirent.h>
#include <thread>


int test_Remove_same_video(void *pvRsvHandle);
void init_rsv(void **pvvrsvHandle)
{
    rsv_open(pvvrsvHandle);
}


int main(int argc,char *argv[])
{
    void *pvRsvHandle;
    //rsv_open(&pvRsvHandle);
    std::thread thd = std::thread(init_rsv, &pvRsvHandle);
    thd.join();

    test_Remove_same_video(pvRsvHandle);
}

void List(char *path, std::vector<std::string> &vecfiles)
{
    struct dirent *ent = NULL;
    DIR *pDir;
    char childpath[1024];
    if((pDir = opendir(path)) != NULL)
    {
        while(NULL != (ent = readdir(pDir)))
        {
            if(ent->d_type == 8)                 // d_type：8-文件，4-目录
            {
                //printf("File:\t%s\n", ent->d_name);
                sprintf(childpath,"%s/%s",path,ent->d_name);
                int lens = strlen(ent->d_name);
                if(lens > 4)
                {
                    std::string str(ent->d_name);
                    //you should add other video type here to support more, as mp4 etc.
                    if(str.substr(lens-4, 4) == ".flv" || str.substr(lens-4, 4) == ".avi")
                        vecfiles.push_back(childpath);
                }

            }
            else if(ent->d_name[0] != '.')
            {
                sprintf(childpath,"%s/%s",path,ent->d_name);
                //printf("path:%s\n",childpath);
                List(childpath, vecfiles);                   // 递归遍历子目录
            }
        }
        closedir(pDir);
    }
    else
        printf("Open Dir-[%s] failed.\n", path);
}

int test_Remove_same_video(void *pvRsvHandle)
{
    char rootpath[] = "/media/d/datasets/shenyang_ting/video";
    std::vector<std::string> vecfiles;
    std::vector<std::string> uuidss;
    List(rootpath, vecfiles);


    //void *pvRsvHandle;
    //rsv_open(&pvRsvHandle);


    int count = vecfiles.size();
    int load_num = 0, same_num = 0;

    double maxtime = 0;

    //test begin
//    std::string videopath1 ="/media/d/datasets/shenyang_ting/video/video-3/2017年凌海市系列盗窃临街商铺案/5b78d1da-ca4a-58ee-5649-83bccbf04428.flv";
//    std::string videopath2 = "/media/d/datasets/shenyang_ting/video/video-2/2017年凌海市系列盗窃临街商铺案/5b78d1da-ca4a-58ee-5649-83bccbf04428.flv";

//    vecfiles.clear();
//    vecfiles.push_back(videopath1);
//    vecfiles.push_back(videopath2);
    //test end

    double totalbegin = static_cast<double>(cvGetTickCount());

    //for(int i=0; i<count; ++i)
    for(int i=0; i<count; ++i)
    {
        std::string uuid;
        std::string samepath;
        double start = static_cast<double>(cvGetTickCount());
        rsv_checkvideo(pvRsvHandle, vecfiles[i], uuid, samepath);
        if(samepath.empty())
        {
            rsv_loadfeature(pvRsvHandle, vecfiles[i], uuid);

            load_num++;
            std::cout << "load success: " << load_num << vecfiles[i] << std::endl;
            //if(load_num % 100 == 0)
                //std::cout << "load success: " << load_num << vecfiles[i] << std::endl;
        }
        else
        {
            same_num++;
            std::cout << "The " << same_num << "th: " << vecfiles[i] << " same as " << samepath << std::endl;
        }
        double time = ((double)cvGetTickCount() - start) / cvGetTickFrequency();
        std::cout << "Time =  " << time/1000 << "ms" << std::endl;
        if(maxtime < time)
            maxtime = time;

    }
    rsv_close(pvRsvHandle);

    double totaltime = ((double)cvGetTickCount() - totalbegin) / cvGetTickFrequency();
    std::cout << "Total video num: " << count << std::endl;
    std::cout << "Load Success num: " << load_num << std::endl;
    std::cout << "Same Video num: " << same_num << std::endl;
    std::cout << "Max Time: " << maxtime/1000 << "ms" << std::endl;
    std::cout << "Total Time: " << totaltime/1000 << "ms" << std::endl;
    return 0;
}
