import json
import os

import numpy as np 
import cv2


DEFAULT_CALIBRATION = {
    "pitch_width": 68.0,
    "pitch_length": 105.0,
    "pixel_vertices": [
        [110, 1035],
        [265, 275],
        [910, 260],
        [1640, 915],
    ],
}

class ViewTransformer():
    def __init__(self, calibration_path=None, video_key=None):
        calibration, calibration_name = self.load_calibration(calibration_path, video_key)

        self.calibration_name = calibration_name
        self.pitch_width = float(calibration["pitch_width"])
        self.pitch_length = float(calibration["pitch_length"])

        self.pixel_vertices = np.array(calibration["pixel_vertices"], dtype=np.float32)
        self.target_vertices = np.array([
            [0, self.pitch_width],
            [0, 0],
            [self.pitch_length, 0],
            [self.pitch_length, self.pitch_width]
        ], dtype=np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def merge_calibration(self, base_calibration, override_calibration):
        merged = {
            "pitch_width": float(override_calibration.get("pitch_width", base_calibration["pitch_width"])),
            "pitch_length": float(override_calibration.get("pitch_length", base_calibration["pitch_length"])),
            "pixel_vertices": override_calibration.get("pixel_vertices", base_calibration["pixel_vertices"]),
        }

        pixel_vertices = np.array(merged["pixel_vertices"], dtype=np.float32)
        if pixel_vertices.shape != (4, 2):
            raise ValueError("Calibration 'pixel_vertices' must contain exactly four [x, y] points.")

        merged["pixel_vertices"] = pixel_vertices.tolist()
        return merged

    def load_calibration(self, calibration_path, video_key):
        default_calibration = self.merge_calibration(DEFAULT_CALIBRATION, {})
        if calibration_path is None or not os.path.exists(calibration_path):
            return default_calibration, "built-in default"

        with open(calibration_path, "r", encoding="utf-8") as calibration_file:
            calibration_data = json.load(calibration_file)

        merged_default = self.merge_calibration(
            default_calibration,
            calibration_data.get("default", {})
        )

        candidate_keys = []
        if video_key:
            candidate_keys.extend([
                video_key,
                os.path.basename(video_key),
                os.path.splitext(os.path.basename(video_key))[0],
            ])

        for candidate_key in candidate_keys:
            if candidate_key in calibration_data:
                return self.merge_calibration(merged_default, calibration_data[candidate_key]), candidate_key

        return merged_default, "default"

    def transform_point(self,point):
        p = (int(point[0]),int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices,p,False) >= 0 
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        transform_point = cv2.perspectiveTransform(reshaped_point,self.perspective_transformer)
        return transform_point.reshape(-1,2)

    def add_transformed_position_to_tracks(self,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    position_transformed = self.transform_point(position)
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed
