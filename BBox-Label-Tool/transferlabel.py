import os
from os import walk
from PIL import Image
from shutil import copyfile

# Function to convert to YOLO format

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

inputpath = "Labels/001/"
outputpath = "YOLO_dataset/"

cls_id = 0

# Get files in the inputpath
mypath=inputpath
txt_name_list = []
for (dirpath, dirnames, filenames) in walk(mypath):
    txt_name_list.extend(filenames)
    break

for txt_name in txt_name_list:
    txt_path = mypath + txt_name
    txt_file = open(txt_path, "r")
    
    lines = txt_file.read().split('\n') 
    
    t = int(lines[0]) # t contains how many bounding boxes in an image
    
    if len(lines) != 2:
        """ Open output text files """
        txt_outpath = outputpath + txt_name
        print("Output:" + txt_outpath)
        txt_outfile = open(txt_outpath, "w")
    
        for i in range(1,t+1):
            line = lines[i].split(" ")
            xmin = line[0]
            xmax = line[2]
            ymin = line[1]
            ymax = line[3]
        
            img_path = "Images/" + "001/" + os.path.splitext(txt_name)[0] + ".jpg"
            im=Image.open(img_path)
            w= int(im.size[0])
            h= int(im.size[1])
            #print(w, h)
            b = (float(xmin), float(xmax), float(ymin), float(ymax))
            bb = convert((w,h), b)
            #print(bb)
            txt_outfile.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + "\n")
        txt_outfile.close()
        dst_path = "YOLO_dataset/" + os.path.splitext(txt_name)[0] + ".jpg"
        copyfile(img_path, dst_path)