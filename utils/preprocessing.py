import cv2
import os
import numpy as np


# -------------------------------
# Frame Extraction
# -------------------------------
def extract_frames(video_path, out_dir, resize=(64, 64)):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frames.append(frame)

    cap.release()
    frames = np.array(frames)  # (T, H, W, C)
    np.save(os.path.join(out_dir, "frames.npy"), frames)
    print(f"✅ Saved {len(frames)} frames to {out_dir}/frames.npy")
    return frames


# -------------------------------
# Edge Detection
# -------------------------------
def compute_edges(frames, out_dir):
    edges = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edge_map = cv2.Canny(gray, 100, 200)
        edges.append(edge_map)

    edges = np.array(edges)  # (T, H, W)
    np.save(os.path.join(out_dir, "edges.npy"), edges)
    print(f"✅ Saved edges to {out_dir}/edges.npy")
    return edges


# -------------------------------
# Color Histograms
# -------------------------------
def compute_histograms(frames, out_dir, bins=16):
    hists = []
    for frame in frames:
        hist_b = cv2.calcHist([frame], [0], None, [bins], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [bins], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [bins], [0, 256])
        hist = np.concatenate([hist_b, hist_g, hist_r]).flatten()
        hist /= hist.sum()  # normalize
        hists.append(hist)

    hists = np.array(hists)  # (T, bins*3)
    np.save(os.path.join(out_dir, "hist.npy"), hists)
    print(f"✅ Saved histograms to {out_dir}/hist.npy")
    return hists


# -------------------------------
# Optical Flow (Farnebäck)
# -------------------------------
def compute_optical_flow(frames, out_dir):
    flows = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )  # (H, W, 2)
        flows.append(flow)
        prev_gray = gray

    flows = np.array(flows)  # (T-1, H, W, 2)
    np.save(os.path.join(out_dir, "flow.npy"), flows)
    print(f"✅ Saved optical flow to {out_dir}/flow.npy")
    return flows


# -------------------------------
# Motion Vectors (Dummy for now)
# -------------------------------
def compute_motion_vectors(frames, out_dir):
    """
    Placeholder motion vectors (random).
    Later: Replace with FFmpeg extraction.
    Shape: (T-1, H//16, W//16, 2) -> block motion vectors
    """
    T, H, W, _ = frames.shape
    motion = np.random.randn(T-1, H//16, W//16, 2).astype(np.float32)
    np.save(os.path.join(out_dir, "motion.npy"), motion)
    print(f"✅ Saved dummy motion vectors to {out_dir}/motion.npy")
    return motion


# -------------------------------
# Coding Features (Dummy for now)
# -------------------------------
def compute_coding_features(frames, out_dir):
    """
    Placeholder coding features per frame.
    Later: Extract QP, frame type, bitrate, etc. using FFmpeg logs.
    Shape: (T, 10) -> e.g., 10 codec stats
    """
    T = frames.shape[0]
    coding = np.random.rand(T, 10).astype(np.float32)
    np.save(os.path.join(out_dir, "coding.npy"), coding)
    print(f"✅ Saved dummy coding features to {out_dir}/coding.npy")
    return coding


# -------------------------------
# Full Pipeline
# -------------------------------
def preprocess_video(video_path, processed_dir="data/processed", resize=(64, 64)):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(processed_dir, video_name)
    os.makedirs(out_dir, exist_ok=True)

    frames = extract_frames(video_path, out_dir, resize)
    edges = compute_edges(frames, out_dir)
    hists = compute_histograms(frames, out_dir)
    flows = compute_optical_flow(frames, out_dir)
    motion = compute_motion_vectors(frames, out_dir)
    coding = compute_coding_features(frames, out_dir)

    print(f"🎉 Preprocessing complete for {video_name}")
    return {
        "frames": frames,
        "edges": edges,
        "hists": hists,
        "flows": flows,
        "motion": motion,
        "coding": coding
    }


if __name__ == "__main__":
    sample_video = "data/raw/movie_clip_01.mp4"
    preprocess_video(sample_video)
