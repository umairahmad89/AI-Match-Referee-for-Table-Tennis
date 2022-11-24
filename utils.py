import cv2
import PIL as pl
import numpy as np
import os
import json
import torch

def coordinates_constructor(set1, set2):
    """
    this function takes two numpy arrays and return an array 
    containing the coordinates constructed using the method
    given in the paper
    """

# [[-0.25691497 -0.10765906]
#  [-0.21242529 -0.08120865]
#  [-0.2257743  -0.10708892]
#  [-0.25207573 -0.08539008]
#  [-0.20851406 -0.07837933]]
# [[0.11667728 0.1695235 ]
#  [0.12115104 0.13122699]
#  [0.08661354 0.16256946]
#  [0.13329902 0.13906519]
#  [0.15259045 0.30198023]]
    resultant_coords = np.zeros_like(set1)
    for i, (p1, p2) in enumerate(zip(set1, set2)):
        p1 = list(p1)
        p2 = list(p2)
        x1,y1 = p1[0],p1[1]
        x2,y2 = p2[0],p2[1]

        x = x1*(1920/320)-(320/2)+x2
        y = y2*(1080/128)-(128/2)+y2
        temp = np.array([x,y])
        resultant_coords[i] = temp
    print(resultant_coords)
            
    return torch.tensor(resultant_coords,dtype=torch.float32)



def normalize_TTNet(set1, set2, dim1, dim2):
    # set1 -> x1, y1
    # set2 -> x2, y2
    # dim1 -> w0, h0
    # dim2 -> w1, h1

    return set1[0] * (dim1[0]/dim2[0]) - (dim2[0]/2) + set2[0], set1[1] * (dim1[1]/dim2[1]) - (dim2[1]/2) + set2[1]



## change name 
def markups_preprocessing(root_dir):

    data = {}
    for folder in os.listdir(root_dir):
        json_file_path = [path for path in os.listdir(os.path.join(root_dir,folder)) if path.endswith(".json")]

        # assumption, there are only two kinds of file/folder
        # --> json and images folder
        folder_name = [name for name in os.listdir(os.path.join(root_dir,folder)) if not name.endswith(".json")][0]
        # we have two json files in json_file_path
        # 
        # 
        # #
        f1 = open(os.path.join(root_dir,folder,json_file_path[0]))
        f2 = open(os.path.join(root_dir,folder,json_file_path[1]))
        json_1 = json.load(f1)
        json_2 = json.load(f2)
        events_markup_file = ""
        ball_markup = ""
        if isinstance(list(json_1.values())[0],str):
            events_markup_file = json_1
            ball_markup = json_2

        else:
            events_markup_file = json_2
            ball_markup = json_1

            

        for id, value in events_markup_file.items():
            # ball markup = "186": {"x": 636, "y": 550}
            # event markup = "190": "bounce"
            if id not in list(ball_markup.keys()):
                ball_markup[id] = {"x":-1,"y":-1}
            
            val = ball_markup[id]

            val["event"] = events_markup_file[id]
            key = os.path.join(root_dir,folder,folder_name,f"{id}.png")
            
            data[key] = val


    return data







def denormalized_points(coords):
    
    x = (coords[0]/320)*1920
    y = (coords[1]/128)*1080
    return (x, y)

def normalize(coords):

    x = (coords[0]/1920)*320
    y = (coords[1]/1080)*128
    return (x, y)
    


def normalized_markups(data, mapping_events):

    """
    it takes a dictionary with the following structure:
    {"x":_,"y":_,"event":_}

    """

    coordinates = normalize((data["x"],data["y"]))
    return {"x":coordinates[0],"y":coordinates[1],\
         "event":mapping_events[data["event"]]}

    







def rectangle_coordinates(coordinates):
    '''
    this function takes a point and return a pair of coordinates which inclosed this point

     top_left(x,y)
     _________________
    |                 |
    |   (x,y)         |
    |                 |
    |                 |
    ```````````````````
                    bottom_right(x,y)
    '''
    w_original = 1920
    h_original = 1080

    w_resize = 320
    h_resize = 128
    
    x_min = max(0, coordinates[0] - int(w_resize / 2))
    y_min = max(0, coordinates[1] - int(h_resize / 2))

    x_max = min(w_original, x_min + w_resize)
    y_max = min(h_original, y_min + h_resize)
    return (x_min,y_min,x_max,y_max)

def resize_around_coords(img, coord=(0,0)):

    # i have hard coded it for the time being
    # will pass dimensions as arguments later
    #
    # ERROR HERE
    # 
    # #
    
    return img.crop(rectangle_coordinates(coord))

def inside_rectangle(top_left, bottom_right, point):
    # all the points are un-normalised
    if (point[0] > top_left[0] and point[1] > top_left[1] \
        and point[0]< bottom_right[0] and point[1] < bottom_right[1]):
        return True
    else:
        return False
    


