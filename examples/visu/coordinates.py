

import os
import json
import magic

from PIL import Image
import cv2
#from pydub import AudioSegment


def list_discoveries(path):
    discoveries = []
    #list all directories in path
    for root, dirs, files in os.walk(path):
        for name in dirs:
            discovery={}

            #load discovery.json file
            with open(os.path.join(root, name, 'discovery.json')) as f:
                discovery_details = json.load(f)
            
            discovery_embedding=discovery_details['output']


            
            #list all files ending with .json
            for file in os.listdir(os.path.join(root, name)):
                if file.endswith(".discovery"):

                            

                    discovery['visual']=os.path.join(root, name, file)
                    mime = magic.Magic(mime=True)
                    mimetype = mime.from_file(discovery['visual'])

                    #if image is detected
                    if 'image' in mimetype:
                        #get img size
                        img = Image.open(discovery['visual'])
                        img_size = img.size
                        discovery['sx']=img_size[0]
                        discovery['sy']=img_size[1]
                    elif 'video' in mimetype:
                        #get video size
                        video = cv2.VideoCapture(discovery['visual'])
                        video_size = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        discovery['sx']=video_size[0]
                        discovery['sy']=video_size[1]


                    discovery['mimetype']=mimetype    
                    discovery['embedding']=discovery_embedding
                    discoveries.append(discovery)

                    break
    return discoveries



import numpy as np
from sklearn.decomposition import PCA

def compute_coordinates(path):
    discoveries = list_discoveries(path)
    X = np.array([discovery['embedding'] for discovery in discoveries
                  
                  
                  ])
    
    #replace all nan with the mean of the column



    pca = PCA(n_components=2)
    pca.fit(X)
    embedding = pca.transform(X)


    for i, discovery in enumerate(discoveries):
        del discovery['embedding']
        discovery['x'] = embedding[i][0].item()
        discovery['y'] = embedding[i][1].item()


    min_x = min(discovery['x'] for discovery in discoveries)
    max_x = max(discovery['x'] for discovery in discoveries)
    min_y = min(discovery['y'] for discovery in discoveries)
    max_y = max(discovery['y'] for discovery in discoveries)

    for discovery in discoveries:
        discovery['x'] = (discovery['x'] - min_x) / (max_x - min_x) - 0.5
        discovery['y'] = (discovery['y'] - min_y) / (max_y - min_y) - 0.5


    #remove path from visual
    for discovery in discoveries:
        discovery['visual'] = discovery['visual'][ len(path):]

    with open('static/discoveries.json', 'w') as f:
        json.dump(discoveries, f)