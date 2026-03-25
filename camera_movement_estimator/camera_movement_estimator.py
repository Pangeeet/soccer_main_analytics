import pickle
import cv2
import numpy as np
import sys
import os
sys.path.append('../')
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self,frame):
        self.minimum_distance = 5
        self.minimum_tracked_features = 8

        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TermCriteria_COUNT, 10, 0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.feature_params = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7
        )

        self.feature_mask = self.build_feature_mask(first_frame_grayscale)

    def build_feature_mask(self, grayscale_frame):
        height, width = grayscale_frame.shape
        mask_features = np.zeros_like(grayscale_frame)

        edge_band_width = max(20, width // 14)
        top_band_height = max(40, height // 5)

        mask_features[:, :edge_band_width] = 1
        mask_features[:, width - edge_band_width:] = 1
        mask_features[:top_band_height, :] = 1

        return mask_features

    def get_tracking_features(self, grayscale_frame):
        if self.feature_mask.shape != grayscale_frame.shape:
            self.feature_mask = self.build_feature_mask(grayscale_frame)

        features = cv2.goodFeaturesToTrack(
            grayscale_frame,
            mask=self.feature_mask,
            **self.feature_params
        )
        if features is None or len(features) == 0:
            features = cv2.goodFeaturesToTrack(
                grayscale_frame,
                mask=None,
                **self.feature_params
            )

        return features

    def add_adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # read the stub
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)

        if len(frames) == 0:
            return []

        camera_movement = [[0,0] for _ in frames]

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = self.get_tracking_features(old_gray)

        for frame_num in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            if old_features is None or len(old_features) == 0:
                old_features = self.get_tracking_features(old_gray)

            if old_features is None or len(old_features) == 0:
                old_gray = frame_gray.copy()
                old_features = self.get_tracking_features(frame_gray)
                continue

            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                old_gray,
                frame_gray,
                old_features,
                None,
                **self.lk_params
            )

            if new_features is None or status is None:
                old_gray = frame_gray.copy()
                old_features = self.get_tracking_features(frame_gray)
                continue

            valid_mask = status.reshape(-1) == 1
            valid_new_features = new_features[valid_mask]
            valid_old_features = old_features[valid_mask]

            if len(valid_new_features) == 0:
                old_gray = frame_gray.copy()
                old_features = self.get_tracking_features(frame_gray)
                continue

            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0

            for new, old in zip(valid_new_features, valid_old_features):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point,old_features_point)
                if distance>max_distance:
                    max_distance = distance
                    camera_movement_x,camera_movement_y = measure_xy_distance(old_features_point, new_features_point ) 
            
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x,camera_movement_y]
                old_features = self.get_tracking_features(frame_gray)
            else:
                old_features = valid_new_features.reshape(-1, 1, 2)
                if len(old_features) < self.minimum_tracked_features:
                    old_features = self.get_tracking_features(frame_gray)

            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)


        return camera_movement
    

    def draw_camera_movement(self, frames, camera_movement_per_frame, copy_frames=False):
        output_frames = [] if copy_frames else frames

        for frame_num, source_frame in enumerate(frames):
            frame = source_frame.copy() if copy_frames else source_frame

            panel_x2 = min(500, frame.shape[1])
            panel_y2 = min(100, frame.shape[0])
            panel_roi = frame[:panel_y2, :panel_x2]
            overlay = panel_roi.copy()
            overlay[:] = (255, 255, 255)
            alpha =0.6
            cv2.addWeighted(overlay,alpha,panel_roi,1-alpha,0,panel_roi)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            if copy_frames:
                output_frames.append(frame)

        return output_frames
