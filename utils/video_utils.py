import cv2


DEFAULT_VIDEO_FPS = 24.0


def read_video(video_path, return_metadata=False):
    cap = cv2.VideoCapture(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frames = []
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        frames.append(frame)

    cap.release()

    metadata = {
        "fps": fps if fps > 0 else DEFAULT_VIDEO_FPS,
        "frame_count": frame_count,
        "frame_width": frame_width,
        "frame_height": frame_height,
    }

    if return_metadata:
        return frames, metadata
    return frames


def save_video(output_video_frames, output_video_path, fps=DEFAULT_VIDEO_FPS):
    if not output_video_frames:
        raise ValueError("No output video frames were provided.")

    target_fps = float(fps) if fps and fps > 0 else DEFAULT_VIDEO_FPS
    frame_width = output_video_frames[0].shape[1]
    frame_height = output_video_frames[0].shape[0]

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, target_fps, (frame_width, frame_height))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
