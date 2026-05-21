import math
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import numpy as np
from sklearn.decomposition import PCA
import cv2
from sklearn.cluster import KMeans
from datetime import datetime

import umap

loaded_json = {}

MIN_STABLE_UMAP_DISCOVERIES = 10
DEFAULT_MAX_RENDERED_DISCOVERIES = 500


def process_discovery(root, name):
    global loaded_json
    discovery = {}

    discovery_path = os.path.join(root, name, 'discovery.json')

    if not os.path.exists(discovery_path):
        return None

    if discovery_path in loaded_json:
        return loaded_json[discovery_path]

    try:
        with open(discovery_path) as f:
            discovery_details = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if 'output' not in discovery_details:
        return None

    try:
        discovery_embedding = np.asarray(discovery_details['output'], dtype=float)
    except (TypeError, ValueError):
        return None

    if np.isnan(discovery_embedding).any():
        print("nan found")
        return None

    # same for infinities
    if np.isinf(discovery_embedding).any():
        print("infinities found")
        return None

    files = os.listdir(os.path.join(root, name))
    # get first file ending with mp4 , only the first one
    mp4_files = [file for file in files if file.endswith('.mp4')]
    if len(mp4_files):
        file = mp4_files[0]

        path = os.path.join(root, name, file)
        discovery['visual'] = path
        discovery['embedding'] = discovery_embedding.tolist()
        loaded_json[discovery_path] = discovery
        return discovery

    png_files = [file for file in files if file.endswith('.png')]
    if len(png_files):
        file = png_files[0]

        path = os.path.join(root, name, file)
        discovery['visual'] = path
        discovery['embedding'] = discovery_embedding.tolist()
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

    print("Number of discoveries: ", len(discoveries))
    return sorted(discoveries, key=lambda discovery: discovery["visual"])


def concatenate_photos(discoveries, output_file='static/concatenated.webm'):
    # make a single photo from all photos
    photos = [cv2.imread(discovery['visual']) for discovery in discoveries]
    concatenated_photo = cv2.hconcat(photos)
    cv2.imwrite(output_file, concatenated_photo)


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
    frame_counts = [int(cv2.VideoCapture(video_path).get(
        cv2.CAP_PROP_FRAME_COUNT)) for video_path in video_paths]
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
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(
        *'VP90'), 5, (total_width, total_height))

    # Initialize frame positions
    frame_positions = [0] * num_videos

    with Pool() as p:
        while True:
            # Prepare arguments for process_frame function
            args = [(i, video_path, frame_positions, frame_counts, black_frame)
                    for i, video_path in enumerate(video_paths)]

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
            grid_frame = np.zeros(
                (total_height, total_width, 3), dtype=np.uint8)

            # Place each frame in the correct position in the grid
            for idx, frame in enumerate(frames):
                row = idx // cols
                col = idx % cols
                y_offset = row * height
                x_offset = col * width
                grid_frame[y_offset:y_offset + height,
                           x_offset:x_offset + width] = frame

            # Write the grid frame to the output video
            out.write(grid_frame)

            # Increment frame positions
            for i in range(num_videos):
                frame_positions[i] += max_frame_count // 20

    out.release()
    cv2.destroyAllWindows()

    return width, height


def export_last_frame(discoveries):
    # Extract a lightweight preview frame for each video to speed up point
    # texture loading in the web UI.
    for discovery in discoveries:
        if not os.path.exists(discovery['visual']):
            continue
        if not discovery['visual'].lower().endswith(".mp4"):
            continue

        img_path = f"{discovery['visual'][:-4]}.jpg"
        if os.path.exists(img_path):
            continue
        video = cv2.VideoCapture(discovery['visual'])
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        ret, frame = video.read()
        video.release()

        if not ret or frame is None:
            continue

        # Keep previews small and compressed since they are only thumbnails for
        # scatter points, not full-fidelity inspection media.
        target_width = 320
        h, w = frame.shape[:2]
        if w > target_width:
            target_height = int(h * (target_width / max(1, w)))
            frame = cv2.resize(frame, (target_width, max(1, target_height)))

        cv2.imwrite(img_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 55])


def _write_json_atomic(path, payload):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def _write_layout_status(static_dir, payload):
    status_path = os.path.join(static_dir, "layout_status.json")
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json_atomic(status_path, payload)


def _valid_discoveries(discoveries):
    filtered = []
    embeddings = []
    expected_dim = None

    for discovery in discoveries:
        if "embedding" not in discovery:
            continue

        embedding = np.asarray(discovery["embedding"], dtype=float)
        if embedding.ndim != 1 or embedding.size == 0:
            continue
        if expected_dim is None:
            expected_dim = embedding.size
        if embedding.size != expected_dim:
            continue
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            continue

        filtered.append(discovery)
        embeddings.append(embedding)

    if not embeddings:
        return [], np.empty((0, 0))

    return filtered, np.vstack(embeddings)


def _normalize_embedding_matrix(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    return (X - mean) / std, mean, std


def _normalize_projection_bounds(embedding):
    min_xy = embedding.min(axis=0)
    max_xy = embedding.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    scale = float(np.max(max_xy - min_xy))
    if scale <= 1e-9:
        scale = 1.0
    return center, scale


def _project_with_temporary_layout(X):
    if len(X) == 1:
        return np.array([[0.0, 0.0]])
    if len(X) == 2:
        return np.array([[-0.5, 0.0], [0.5, 0.0]])

    X_norm, _, _ = _normalize_embedding_matrix(X)
    reducer = PCA(n_components=2, random_state=0)
    embedding = reducer.fit_transform(X_norm)
    center, scale = _normalize_projection_bounds(embedding)
    return (embedding - center) / scale


def _project_with_umap(X):
    X_norm, _, _ = _normalize_embedding_matrix(X)
    reducer = umap.UMAP(
        n_components=2,
        random_state=0,
        n_neighbors=min(10, len(X_norm) - 1),
    )
    embedding = reducer.fit_transform(X_norm)
    center, scale = _normalize_projection_bounds(embedding)
    return (embedding - center) / scale


def _downsample_for_display(discoveries, embedding, max_displayed):
    if len(discoveries) <= max_displayed:
        return discoveries, embedding

    kmeans = KMeans(n_clusters=max_displayed, random_state=0)
    labels = kmeans.fit_predict(embedding)
    centers = kmeans.cluster_centers_

    selected_indices = []
    for cluster_idx, center in enumerate(centers):
        members = np.where(labels == cluster_idx)[0]
        if len(members) == 0:
            continue

        cluster_points = embedding[members]
        nearest_member = members[np.argmin(
            np.linalg.norm(cluster_points - center, axis=1))]
        selected_indices.append(nearest_member)

    selected_indices.sort()
    return [discoveries[i] for i in selected_indices], embedding[selected_indices]


def _saved_coordinates(discoveries, embedding, root_path):
    saved_coordinates = []
    for discovery, point in zip(discoveries, embedding):
        if np.isnan(point).any() or np.isinf(point).any():
            continue
        saved_coordinates.append({
            "x": float(point[0]),
            "y": float(point[1]),
            "visual": discovery["visual"][1 + len(root_path):],
        })
    return saved_coordinates


def compute_coordinates(
    path,
    static_dir='static',
    max_displayed=DEFAULT_MAX_RENDERED_DISCOVERIES,
):
    print("computing coordinates", path)
    discoveries = list_discoveries(path)
    static_discoveries_path = os.path.join(static_dir, 'discoveries.json')
    static_concatenated_path = os.path.join(static_dir, 'concatenated.webm')

    if len(discoveries) == 0:
        _write_json_atomic(static_discoveries_path, [])
        _write_layout_status(static_dir, {
            "mode": "empty",
            "stable": False,
            "count": 0,
            "displayed_count": 0,
            "fit_count": 0,
        })
        if os.path.exists(static_concatenated_path):
            os.remove(static_concatenated_path)
        return

    discoveries, X = _valid_discoveries(discoveries)
    if len(discoveries) == 0:
        _write_json_atomic(static_discoveries_path, [])
        _write_layout_status(static_dir, {
            "mode": "empty",
            "stable": False,
            "count": 0,
            "displayed_count": 0,
            "fit_count": 0,
        })
        return

    if len(discoveries) >= MIN_STABLE_UMAP_DISCOVERIES:
        embedding = _project_with_umap(X)
        layout_mode = "refit_umap"
        stable = False
        fit_count = len(discoveries)
    else:
        embedding = _project_with_temporary_layout(X)
        layout_mode = "bootstrap_pca"
        stable = False
        fit_count = 0

    # width, height=concatenate_videos(discoveries)
    export_last_frame(discoveries)

    display_discoveries, display_embedding = _downsample_for_display(
        discoveries,
        embedding,
        max_displayed,
    )
    saved_coordinates = _saved_coordinates(display_discoveries, display_embedding, path)

    _write_json_atomic(static_discoveries_path, saved_coordinates)
    _write_layout_status(static_dir, {
        "mode": layout_mode,
        "stable": stable,
        "count": len(discoveries),
        "displayed_count": len(saved_coordinates),
        "fit_count": fit_count,
        "max_displayed": max_displayed,
    })

    return None
