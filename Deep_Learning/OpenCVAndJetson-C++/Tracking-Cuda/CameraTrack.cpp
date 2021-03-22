#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/videoio/registry.hpp"

#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/core/cuda.hpp"

#include <iostream>
using namespace std;
using namespace cv;

string cascadeName = "data/haarcascade_frontalface_alt.xml";


Ptr<cuda::CascadeClassifier> cascade;
VideoCapture webcam;
//Ptr<cv::cuda::HOG> gpu_hog;

cuda::GpuMat gray, gimg, gfaces;

vector<Rect> faces;

       int wd=640; //Window dimensio
       int ht=480; //Window dimension
//        int wd=320; // 320
//        int ht=240; // 240


void info()
{
    cout << "Gpu active: " << cuda::getCudaEnabledDeviceCount() << endl;
    cuda::setDevice(0);
    cuda::DeviceInfo dev;
    cout << "Gpu name: " << dev.name() << endl;
//    cout << "Processors number: " << dev.multiProcessorCount() << endl;
//    cout << "Total memory (MB): " << dev.totalMemory()/1000000 << endl;
}

int verify()
{
    int camera = 0;
    if(!webcam.open("/dev/video0", CAP_V4L2))
    {
        cout << "Webcam from camera #" <<  camera << " didn't work" << endl;
        return 1;
    }

    webcam.set(CAP_PROP_FRAME_WIDTH,wd);
    webcam.set(CAP_PROP_FRAME_HEIGHT,ht);
    webcam.set(CAP_OPENNI_QVGA_30HZ, 30);
    webcam.set(CAP_PROP_MODE, 1);
    webcam.set(CAP_PROP_FPS,30);
    cout << "Fps: " << webcam.get(CAP_PROP_FPS) << endl;
    cout << "Mode: " << webcam.get(CAP_PROP_MODE) << endl;
//    for (VideoCaptureAPIs b : cv::videoio_registry::getCameraBackends()) {cout << cv::videoio_registry::getBackendName(b) << endl;}

    cascade = cuda::CascadeClassifier::create(cascadeName);

    if (cascade == NULL)
    {
        cerr << "ERROR: Could not load classifier cascade: " << cascadeName << endl;
        return -1;
    }
    cascade->setMaxNumObjects(1);
    cascade->setScaleFactor(1.1);
    cascade->setMinNeighbors(10);
    cascade->setFindLargestObject(true);

    return 0;
}

bool first=true;
float wt2, ht2, cx, cy;
float wt2m, ht2m, cxm, cym;
float g=0.1;
Scalar color = Scalar(255,0,0);

void detect(Mat& img)
{
        gimg.upload(img);

        cuda::cvtColor( gimg, gray, COLOR_BGR2GRAY );

        cascade->detectMultiScale(gray, gfaces);
        cascade->convert(gfaces, faces);

        for ( size_t i = 0; i < faces.size(); i++ )
        {
            Rect r = faces[i];
            wt2=r.width/2;
            ht2=r.height/2;
            cx=r.x+wt2;
            cy=r.y+ht2;
            if (wt2==0) first=true;
            if (first) {wt2m=wt2; ht2m=ht2; cxm=cx; cym=cy;}
            else{wt2m=wt2m+g*(wt2-wt2m); ht2m=ht2m+g*(ht2-ht2m); cxm=cxm+g*(cx-cxm); cym=cym+g*(cy-cym);}
            rectangle(img, Point(cvRound(cxm-wt2m),cvRound(cym-ht2m)), Point(cvRound(cxm+wt2m), cvRound(cym+ht2m)),color, 2, 8, 0);
            first=false;
        }
}


int main( int argc, const char** argv )
{
    Mat frame;
    double t = 0;
    info();
    if (verify()>0) return 1;
    char *winname="Tracking";
    namedWindow(winname, WINDOW_AUTOSIZE);


    if( webcam.isOpened() )
    {
        cout << "Video capturing " << wd << "x" << ht << " has been started ..." << endl;
        t = (double)getTickCount();
        int nf=0;
        while (waitKey(1) < 0)
        {
            if (!webcam.read(frame)) break;
            detect(frame);
            imshow(winname,frame);
            nf++;
            if (nf==100)
                {
                 t = (double)getTickCount() - t;
                 double tf=t/(100*getTickFrequency());
                 printf("Detection(Frame) time = %g s\n", tf);
                 printf("Frame/s : %g\n",1/tf);
                 t = (double)getTickCount();
                 nf=0;
                }
        }
    }


}

// 640x480 16f/s nodetect 30f/s 30f/s (direct no ssh)
// 320x240 27f/s nodetect 30f/s 30f/s (direct no ssh)
