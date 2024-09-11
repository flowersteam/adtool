

import os
import json
import numpy as np
from sklearn.decomposition import PCA
import cv2
from sklearn.cluster import KMeans
import umap


#from pydub import AudioSegment


loaded_json={

}


import os
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_discovery(root, name):
    global loaded_json
    discovery = {}


    discovery_path = os.path.join(root, name, 'discovery.json')

    if not os.path.exists(discovery_path):
        return None

    if discovery_path in loaded_json:
        return loaded_json[discovery_path]

    with open(discovery_path) as f:
        discovery_details = json.load(f)

    discovery_embedding = discovery_details['output']

    if np.isnan(discovery_embedding).any():
        print("nan found")
        return None
    # same for None in embedding
    if None in discovery_embedding:
        print("None found")
        return None
    
    # same for infinities
    if np.isinf(discovery_embedding).any():
        print("infinities found")
        return None

    for file in os.listdir(os.path.join(root, name)):
        path = os.path.join(root, name, file)
        if file.endswith('.mp4'):
            discovery['visual'] = path
            discovery['embedding'] = discovery_embedding
            loaded_json[discovery_path] = discovery
            return discovery

    return None

def list_discoveries(path):
    discoveries = []
    tasks = []
    global loaded_json

    with ThreadPoolExecutor() as executor:
        for root, dirs, _ in os.walk(path):
            for name in dirs:
                tasks.append(executor.submit(process_discovery, root, name))

        for future in as_completed(tasks):
            result = future.result()
            if result:
                discoveries.append(result)

    return discoveries

import cv2
import numpy as np
from multiprocessing import Pool

def concatenate_photos(discoveries, output_file='static/concatenated.webm'):
    #make a single photo from all photos
    photos = [cv2.imread(discovery['visual']) for discovery in discoveries]
    concatenated_photo = cv2.hconcat(photos)
    cv2.imwrite(output_file, concatenated_photo)


import cv2
import numpy as np
from multiprocessing import Pool
import math

def process_frame(args):
    i, video_path, frame_positions, frame_counts, black_frame = args
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_positions[i])
    ret, frame = video.read()
    video.release()
    if frame_positions[i] >= frame_counts[i]:
        frame = black_frame
    return i, frame

def concatenate_videos(discoveries, output_file='static/concatenated.webm'):
    video_paths = [discovery['visual'] for discovery in discoveries]

    # Get the width and height of the first video
    video = cv2.VideoCapture(video_paths[0])
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()

    # Calculate the number of frames in each video
    frame_counts = [int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)) for video_path in video_paths]
    max_frame_count = max(frame_counts)

    # Calculate the number of rows and columns needed to form a grid
    num_videos = len(video_paths)
    rows = math.ceil(math.sqrt(num_videos))
    cols = math.ceil(num_videos / rows)

    # Create a black frame with the same size as the video frame
    black_frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Create a VideoWriter object with the output file name, fourcc code, frames per second, and frame size
    total_width = width * cols
    total_height = height * rows
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'VP90'), 5, (total_width, total_height))

    # Initialize frame positions
    frame_positions = [0] * num_videos

    with Pool() as p:
        while True:
            # Prepare arguments for process_frame function
            args = [(i, video_path, frame_positions, frame_counts, black_frame) for i, video_path in enumerate(video_paths)]

            # Process frames in parallel
            results = p.map(process_frame, args)

            # Break the loop if all videos are finished
            if all(frame_positions[i] >= frame_counts[i] for i in range(num_videos)):
                print("Info: All videos have been processed")
                break

            # Sort the results by video index
            results.sort(key=lambda x: x[0])

            # Extract the frames
            frames = [result[1] for result in results]

            # Create an empty frame for the grid
            grid_frame = np.zeros((total_height, total_width, 3), dtype=np.uint8)

            # Place each frame in the correct position in the grid
            for idx, frame in enumerate(frames):
                row = idx // cols
                col = idx % cols
                y_offset = row * height
                x_offset = col * width
                grid_frame[y_offset:y_offset + height, x_offset:x_offset + width] = frame

            # Write the grid frame to the output video
            out.write(grid_frame)

            # Increment frame positions
            for i in range(num_videos):
                frame_positions[i] += max_frame_count // 20

    out.release()
    cv2.destroyAllWindows()

    return width, height


def export_last_frame(discoveries):
    # extract last frame of each video and save them in respective folder
    for discovery in discoveries:
        if not os.path.exists(discovery['visual']):
            continue

        img_path = f"{discovery['visual'][:-4]}.png"
        if os.path.exists(img_path):
            continue
        video = cv2.VideoCapture(discovery['visual'])
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = video.read()
        video.release()

        # replace .mp4 with .jpg
        cv2.imwrite(img_path, frame)

def compute_coordinates(path):
    global pca
    print("computing coordinates", path)
    discoveries = list_discoveries(path)
    if len(discoveries) == 0:
        #touch discoveries.json
        with open('static/discoveries.json', 'w') as f:
            f.write('[]')
            # rm static/concatenated.webm if exists
        if os.path.exists('static/concatenated.webm'):
            os.remove('static/concatenated.webm')

        return

    # if less than 2 discoveries, return
    if len(discoveries) == 0:
        return
    if len(discoveries) <3:
        with open('static/discoveries.json', 'w') as f:
            json.dump([{
                'x': 0,
                'y': 0,
                'visual': discoveries[0]['visual'][len(path):]
            }], f)
        return
    
    if len(discoveries) == 2:
        with open('static/discoveries.json', 'w') as f:
            json.dump([{
                'x': -1,
                'y': 0,
                'visual': discoveries[0]['visual'][len(path):]
            }, {
                'x': 1,
                'y': 0,
                'visual': discoveries[1]['visual'][len(path):]
            }], f)
        return
            

        
    X = np.array([discovery['embedding'] for discovery in discoveries  if 'embedding' in discovery])




    if len(discoveries) > 1000:
        # keep only the top 100 most disctinct  discoveries 
        kmeans = KMeans(n_clusters=1000, random_state=0)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        top_discoveries=[]
        for center in centers:
            min_distance=float('inf')
            top_discovery=None
            for discovery in discoveries:
                distance=np.linalg.norm(discovery['embedding']-center)
                if distance<min_distance and discovery not in top_discoveries:
                    min_distance=distance
                    top_discovery=discovery
            top_discoveries.append(top_discovery)

        discoveries=top_discoveries



    # check if there is nan values
    if np.isnan(X).any():
        print("nan found in X")
        return
    
    # remove enries with nan

    



    
    


    # normalize X
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    




   # pca = PCA(n_components=2, random_state=0, whiten=True)
    pca = umap.UMAP(n_components=2, random_state=0, n_neighbors=min(10, len(X)))    
    pca.fit(X)



    embedding = pca.transform(X)
    # check if there is nan values
    if np.isnan(embedding).any():
        print("nan found in embedding")

    saved_coordinates = []



    for i, discovery in enumerate(discoveries):
        # if nan in embedding, skip
        if np.isnan(embedding[i]).any():
            continue
        saved_coordinates.append({
            'x': embedding[i][0].item(),
            'y': embedding[i][1].item(),
            'visual': discovery['visual']
        })





    min_x = min(discovery['x'] for discovery in saved_coordinates)
    max_x = max(discovery['x'] for discovery in saved_coordinates)
    min_y = min(discovery['y'] for discovery in saved_coordinates)
    max_y = max(discovery['y'] for discovery in saved_coordinates)

    for discovery in saved_coordinates:
        discovery['x'] = (discovery['x'] - min_x) / (max_x - min_x) - 0.5
        discovery['y'] = (discovery['y'] - min_y) / (max_y - min_y) - 0.5

    # same for target
    if os.path.exists(f'{path}/target.json'):
        with open(f'{path}/target.json') as f:
            target = json.load(f)

        target_embedding = pca.transform([target['target']])
        target['x'] = target_embedding[0][0].item()
        target['y'] = target_embedding[0][1].item()

        target['x'] = (target['x'] - min_x) / (max_x - min_x) - 0.5
        target['y'] = (target['y'] - min_y) / (max_y - min_y) - 0.5

        # rewrite target.json
        with open(f"{path}/target.json", "w") as f:
            json.dump(target, f)

   # width, height=concatenate_videos(discoveries)
    export_last_frame(discoveries)

    #remove path from visual
    for discovery in saved_coordinates:
        discovery['visual'] = discovery['visual'][1+ len(path):]
        # discovery['width'] = width
        # discovery['height'] = height

    with open('static/discoveries.json', 'w') as f:
        json.dump(saved_coordinates, f)

    return pca