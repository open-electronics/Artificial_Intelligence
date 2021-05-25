### Yolo on photos 

Yolo deep learning objects detect using OpenCV as backend.
This program is dedicated to verify yolo model on photo. (yolo model default: yolo4). OpenCV load network, adapt image from file to frame dimension used for training and detect objects using labels listed in dedicated file.

This script can be used on Window or Jetson, but you have to change comment in these line

`root = "D:\\"    #Windows
#root = "/"        #Linux
prepath = os.path.join(root, "AIData")        # ex. windows
#prepath = os.path.join(root, "home", "jetson", "AIData")     # ex. linux`

That means that configuration files and weights  have to be positioned in:

- D:\\\AIData\\darknet   (Windows)
- /home/jetson/AIData/darknet  (Jetson)

You can change model (network structure). Weights of pre-trained model are defined by a file with similar name.
Labels file is coco.names except for yolov3-tiny-voc model (in this case is voc.names)

If you don't insert photo-file name as running parameter --photo (Ex. --photo mypicture.jpg) the program start a chooser file window dialog.
Use -h parameter to display help.

`usage: OpencvYoloPhoto.py [-h] [--model MODEL] [--photo PHOTO] [--thresh THRESH] [--linethick LINETHICK]  [--textdim TEXTDIM] [-?]`

`OpenCV-YOLO program`

`optional arguments:`
  `-h, --help            show this help message and exit`
  `--model MODEL         net model path (def: yolov4)`
  `--photo PHOTO         photo file path (def: None)`
  `--thresh THRESH       confidence threshold (def: 0.5)`
  `--linethick LINETHICK`
                        `thickess of rectangle line (def: 1)`
  `--textdim TEXTDIM     label text dimension (def: 0.2)`
  `-?                    Help as -h`