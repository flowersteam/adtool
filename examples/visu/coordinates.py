

import os
import json
import numpy as np
from sklearn.decomposition import PCA
import cv2

#from pydub import AudioSegment


def list_discoveries(path):
    discoveries = []
    #list all directories in path
    for root, dirs, files in os.walk(path):
        for name in dirs:
            discovery={}

            #load discovery.json file

            #check if discovery.json exists
            if not os.path.exists(os.path.join(root, name, 'discovery.json')):
                continue

            with open(os.path.join(root, name, 'discovery.json')) as f:
                discovery_details = json.load(f)
            
            discovery_embedding=discovery_details['output']
            #if contains nan, continue
            if np.isnan(discovery_embedding).any():
                print("nan found")
                continue


            
            #list all files ending with .json
            for file in os.listdir(os.path.join(root, name)):
           #     mime = magic.Magic(mime=True)
                path=os.path.join(root, name, file)
            #    mimetype = mime.from_file(path)

                if file.endswith('.mp4'):
                    #get dimensions of video
                    cap = cv2.VideoCapture(path)
                    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                    discovery['width']=width
                    discovery['height']=height
                    discovery['visual']=path
                    discovery['embedding']=discovery_embedding
                    discoveries.append(discovery)
                    break

                # if file.endswith('.png'):
                #     discovery['width']=discovery_details['width']
                #     discovery['height']=discovery_details['height']
                #     discovery['visual']=path
                #     discovery['embedding']=discovery_embedding
                #     discoveries.append(discovery)
                #     break

    return discoveries


import cv2
import numpy as np
from multiprocessing import Pool

def process_frame(args):
    i, video_path, frame_positions, frame_counts, black_frame = args
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_positions[i])
    ret, frame = video.read()
    video.release()
    if frame_positions[i] >= frame_counts[i]:
        frame = black_frame
    return i, frame

def concatenate_photos(discoveries, output_file='static/concatenated.png'):
    #make a single photo from all photos
    photos = [cv2.imread(discovery['visual']) for discovery in discoveries]
    concatenated_photo = cv2.hconcat(photos)
    cv2.imwrite(output_file, concatenated_photo)


def concatenate_videos(discoveries, output_file='static/concatenated.webm'):
    discoveries=discoveries[:100]
    video_paths = [discovery['visual'] for discovery in discoveries]

    # Get the width and height of the first video
    video = cv2.VideoCapture(video_paths[0])
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()

    # Calculate the total width of the output video
    total_width = width * len(video_paths)

    # Create a black frame with the same size as the video frame
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Get the number of frames in each video
    frame_counts = [int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)) for video_path in video_paths]

    # Create a VideoWriter object with the output file name, fourcc code, frames per second, and frame size
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'VP90'), 5, (total_width, height))

    # Initialize frame positions
    frame_positions = [0] * len(video_paths)

    with Pool() as p:
        while True:
            # Prepare arguments for process_frame function
            args = [(i, video_path, frame_positions, frame_counts, black_frame) for i, video_path in enumerate(video_paths)]

            # Process frames in parallel
            results = p.map(process_frame, args)

            # Break the loop if all videos are finished
            if all(frame_positions[i] >= frame_counts[i] for i in range(len(video_paths))):
                print("Info: All videos have been processed")
                break

            # Sort the results by video index
            results.sort(key=lambda x: x[0])

            # Extract the frames
            frames = [result[1] for result in results]

            # Concatenate the frames horizontally
            concatenated_frame = cv2.hconcat(frames)

            # Write the concatenated frame to the output video
            out.write(concatenated_frame)

            # Increment frame positions
            for i in range(len(video_paths)):
                frame_positions[i] += 3

    out.release()

    cv2.destroyAllWindows()


def compute_coordinates(path):
    discoveries = list_discoveries(path)
    if len(discoveries) == 0:
        #touch discoveries.json
        with open('static/discoveries.json', 'w') as f:
            f.write('[]')
            # rm static/concatenated.webm if exists
        if os.path.exists('static/concatenated.webm'):
            os.remove('static/concatenated.webm')

        return
    
    concatenate_videos(discoveries)
    print("videos concatenated")
    # if less than 2 discoveries, return
    if len(discoveries) < 2:
        return
    X = np.array([discovery['embedding'] for discovery in discoveries  ])
    
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