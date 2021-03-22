#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include <iostream>
using namespace std;
using namespace cv;

string cascadeName = "data/haarcascade_frontalface_alt.xml";


CascadeClassifier cascade;
VideoCapture webcam;

vector<Rect> faces;

        int wd=640; //window width 640
        int ht=480; //window hight 480
//        int wd=320; //320
//        int ht=240; //240

int verify()
{
    int camera = 1;
    if(!webcam.open(camera))
    {
        cout << "Webcam from camera #" <<  camera << " didn't work" << endl;
        return 1;
    }
    webcam.set(CAP_PROP_FRAME_WIDTH,wd);
    webcam.set(CAP_PROP_FRAME_HEIGHT,ht);

    if (!cascade.load(cascadeName))
    {
        cerr << "ERROR: Could not load classifier cascade: " << cascadeName << endl;
        return -1;
    }

    return 0;
}


void detect(Mat& img)
{
        Mat gray;
        cvtColor( img, gray, COLOR_BGR2GRAY );
        cascade.detectMultiScale( gray, faces);
        Scalar color = Scalar(255,0,0);
        for ( size_t i = 0; i < faces.size(); i++ )
        {
            Rect r = faces[i];
            rectangle( img, Point(cvRound(r.x), cvRound(r.y)),
                       Point(cvRound((r.x + r.width-1)), cvRound((r.y + r.height-1))),
                       color, 2, 8, 0);
        }
}


int main( int argc, const char** argv )
{
    Mat frame, image;
    double t = 0;
    if (verify()>0) return 1;

//    cout << getBuildInformation() << endl; exit(0);

    if( webcam.isOpened() )
    {
        cout << "Video capturing " << wd << "x" << ht << " has been started ..." << endl;
        t = (double)getTickCount();
        int nf=0;
        for(;;)
        {
            webcam >> frame;
            if( frame.empty() )  break;
            detect(frame);
            imshow("Frame",frame);
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

            char c = (char)waitKey(1);
            if( c == 27 || c == 'q' || c == 'Q' )
                break;

        }
    }


}

// PC
// 640x480 : 25 f/s
// 320x240 : 30 f/s
// Jetson
// 640x480 : 7 f/s
// 320x240 : 17 f/s