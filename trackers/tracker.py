from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
sys.path.append('../')

from utils import get_center_of_bbox, get_bbox_width, get_foot_position, measure_distance
import cv2
import numpy as np
import pandas as pd


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.min_ball_confidence = 0.12
        self.max_ball_gap_fill = 4
        self.max_ball_area_ratio = 0.0020
        self.ball_edge_margin = 12
        self.player_id_max_gap_frames = 12
        self.player_id_base_match_distance = 1.6
        self.player_id_distance_growth = 0.32
        self.player_id_max_match_distance = 4.8
        self.player_id_min_size_ratio = 0.55
        self.player_id_max_size_ratio = 1.85

    def add_position_to_tracks(self, tracks):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object_name == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object_name][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        original_ball_positions = ball_positions
        ball_bboxes = [
            x.get(1, {}).get('bbox', [np.nan, np.nan, np.nan, np.nan])
            for x in original_ball_positions
        ]
        df_ball_positions = pd.DataFrame(ball_bboxes, columns=['x1', 'y1', 'x2', 'y2'], dtype=float)

        df_ball_positions = df_ball_positions.interpolate(
            limit=self.max_ball_gap_fill,
            limit_direction='both',
            limit_area='inside'
        )

        valid_rows = ~df_ball_positions.isna().any(axis=1)
        smoothed_positions = df_ball_positions.rolling(window=3, min_periods=1, center=True).mean()

        ball_positions = []
        for row_idx, bbox in enumerate(smoothed_positions.to_numpy().tolist()):
            if not valid_rows.iloc[row_idx] or any(pd.isna(value) for value in bbox):
                ball_positions.append({})
            else:
                original_ball_info = dict(original_ball_positions[row_idx].get(1, {}))
                if "bbox" in original_ball_info:
                    original_ball_info["bbox"] = bbox
                    ball_positions.append({1: original_ball_info})
                else:
                    ball_positions.append({
                        1: {
                            "bbox": bbox,
                            "source": "interpolated",
                            "confidence": 0.0,
                        }
                    })

        return ball_positions

    def get_bbox_area(self, bbox):
        width = max(0.0, bbox[2] - bbox[0])
        height = max(0.0, bbox[3] - bbox[1])
        return width * height

    def get_ball_candidate_context(self, center, player_bboxes):
        nearest_player_distance = None
        inside_player_body = False
        inside_player_feet_zone = False

        for player_bbox in player_bboxes:
            foot_distance = measure_distance(get_foot_position(player_bbox), center)
            if nearest_player_distance is None or foot_distance < nearest_player_distance:
                nearest_player_distance = foot_distance

            x1, y1, x2, y2 = player_bbox
            if x1 <= center[0] <= x2 and y1 <= center[1] <= y2:
                height = max(1.0, y2 - y1)
                knee_line = y1 + (0.68 * height)
                if center[1] < knee_line:
                    inside_player_body = True
                else:
                    inside_player_feet_zone = True

        return {
            "nearest_player_distance": nearest_player_distance,
            "inside_player_body": inside_player_body,
            "inside_player_feet_zone": inside_player_feet_zone,
        }

    def is_valid_ball_candidate(self, bbox, frame_shape):
        frame_height, frame_width = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height

        if width <= 1 or height <= 1:
            return False

        if width > frame_width * 0.045 or height > frame_height * 0.06:
            return False

        if area > (frame_width * frame_height * self.max_ball_area_ratio):
            return False

        if x2 < 0 or y2 < 0 or x1 > frame_width or y1 > frame_height:
            return False

        aspect_ratio = width / max(height, 1e-6)
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            return False

        return True

    def shift_bbox(self, bbox, velocity):
        if bbox is None:
            return None

        dx, dy = velocity
        x1, y1, x2, y2 = bbox
        return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

    def get_predicted_ball_center(self, previous_ball_center, previous_ball_velocity, missing_frames):
        if previous_ball_center is None:
            return None

        scale = min(max(missing_frames, 1), 3)
        return (
            previous_ball_center[0] + (previous_ball_velocity[0] * scale),
            previous_ball_center[1] + (previous_ball_velocity[1] * scale)
        )

    def score_ball_candidate(
        self,
        candidate,
        predicted_ball_center,
        previous_ball_bbox,
        previous_ball_velocity,
        missing_frames,
        player_bboxes,
        frame_shape
    ):
        bbox, confidence, center = candidate
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        frame_height, frame_width = frame_shape[:2]

        score = confidence * 180.0

        aspect_ratio = width / max(height, 1e-6)
        score -= abs(1.0 - aspect_ratio) * 18.0

        speed = float(np.linalg.norm(previous_ball_velocity))
        max_jump_distance = 90.0 + (35.0 * min(missing_frames, 3)) + (0.5 * speed)

        if predicted_ball_center is not None:
            distance = measure_distance(center, predicted_ball_center)
            score -= distance * 1.4
            if distance > max_jump_distance:
                score -= 220.0

        if previous_ball_bbox is not None:
            previous_width = previous_ball_bbox[2] - previous_ball_bbox[0]
            previous_height = previous_ball_bbox[3] - previous_ball_bbox[1]
            previous_area = previous_width * previous_height
            score -= abs(area - previous_area) * 0.2

        candidate_context = self.get_ball_candidate_context(center, player_bboxes)
        nearest_player_distance = candidate_context["nearest_player_distance"]
        if nearest_player_distance is not None:
            if nearest_player_distance < 140:
                score += max(0.0, 55.0 - (0.28 * nearest_player_distance))
            else:
                score -= min(45.0, (nearest_player_distance - 140.0) * 0.12)

        if candidate_context["inside_player_body"]:
            score -= 150.0
        elif candidate_context["inside_player_feet_zone"]:
            score += 22.0

        edge_distance = min(center[0], center[1], frame_width - center[0], frame_height - center[1])
        if edge_distance < self.ball_edge_margin:
            score -= (self.ball_edge_margin - edge_distance) * 3.5

        if previous_ball_bbox is None and confidence < 0.18 and nearest_player_distance is not None and nearest_player_distance > 170:
            score -= 25.0

        return score

    def get_min_ball_candidate_score(self, previous_ball_bbox, missing_frames):
        if previous_ball_bbox is None:
            return -18.0
        if missing_frames == 0:
            return -4.0
        if missing_frames == 1:
            return -12.0
        return -20.0

    def build_ball_track_info(self, bbox, source, confidence=0.0, score=None):
        track_info = {
            "bbox": bbox,
            "source": source,
            "confidence": float(confidence),
        }
        if score is not None:
            track_info["score"] = float(score)
        return track_info

    def normalize_track_id(self, track_id):
        try:
            return int(track_id)
        except (TypeError, ValueError):
            return track_id

    def get_player_tracking_point(self, track_info):
        for key in ("position_transformed", "position_adjusted", "position"):
            point = track_info.get(key)
            if point is None:
                continue

            point_array = np.asarray(point, dtype=float).reshape(-1)
            if point_array.size >= 2 and np.isfinite(point_array[:2]).all():
                return point_array[:2]

        bbox = track_info.get("bbox")
        if bbox is None:
            return None

        foot_position = get_foot_position(bbox)
        if foot_position is None:
            return None

        point_array = np.asarray(foot_position, dtype=float).reshape(-1)
        if point_array.size < 2 or not np.isfinite(point_array[:2]).all():
            return None

        return point_array[:2]

    def compute_bbox_iou(self, bbox_a, bbox_b):
        if bbox_a is None or bbox_b is None:
            return 0.0

        x_left = max(float(bbox_a[0]), float(bbox_b[0]))
        y_top = max(float(bbox_a[1]), float(bbox_b[1]))
        x_right = min(float(bbox_a[2]), float(bbox_b[2]))
        y_bottom = min(float(bbox_a[3]), float(bbox_b[3]))

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area_a = self.get_bbox_area(bbox_a)
        area_b = self.get_bbox_area(bbox_b)
        union = max(area_a + area_b - intersection, 1e-6)
        return float(intersection / union)

    def get_bbox_height_ratio(self, current_bbox, previous_bbox):
        if current_bbox is None or previous_bbox is None:
            return 1.0

        current_height = max(1.0, float(current_bbox[3]) - float(current_bbox[1]))
        previous_height = max(1.0, float(previous_bbox[3]) - float(previous_bbox[1]))
        return current_height / previous_height

    def update_stable_player_state(self, stable_states, stable_id, frame_num, raw_track_id, track_info):
        track_point = self.get_player_tracking_point(track_info)
        previous_state = stable_states.get(stable_id)
        previous_point = previous_state["point"] if previous_state is not None else None

        velocity = np.zeros(2, dtype=float)
        if track_point is not None and previous_point is not None:
            velocity = track_point - previous_point
        elif previous_state is not None:
            velocity = previous_state.get("velocity", np.zeros(2, dtype=float))

        stable_states[stable_id] = {
            "frame_num": frame_num,
            "raw_track_id": raw_track_id,
            "point": track_point,
            "bbox": track_info.get("bbox"),
            "velocity": velocity,
        }

    def find_best_stable_player_match(
        self,
        stable_states,
        frame_num,
        current_track_info,
        assigned_stable_ids
    ):
        current_point = self.get_player_tracking_point(current_track_info)
        if current_point is None:
            return None

        current_bbox = current_track_info.get("bbox")
        best_match_id = None
        best_score = None

        for stable_id, state in stable_states.items():
            if stable_id in assigned_stable_ids:
                continue

            previous_frame = state["frame_num"]
            frame_gap = frame_num - previous_frame
            if frame_gap <= 0 or frame_gap > self.player_id_max_gap_frames:
                continue

            previous_point = state.get("point")
            if previous_point is None:
                continue

            previous_velocity = state.get("velocity", np.zeros(2, dtype=float))
            prediction_steps = min(frame_gap, 3)
            predicted_point = previous_point + (previous_velocity * prediction_steps)
            distance = float(np.linalg.norm(current_point - predicted_point))

            max_distance = min(
                self.player_id_max_match_distance,
                self.player_id_base_match_distance + (self.player_id_distance_growth * frame_gap)
            )
            if distance > max_distance:
                continue

            size_ratio = self.get_bbox_height_ratio(current_bbox, state.get("bbox"))
            if size_ratio < self.player_id_min_size_ratio or size_ratio > self.player_id_max_size_ratio:
                continue

            bbox_iou = self.compute_bbox_iou(current_bbox, state.get("bbox"))
            if frame_gap <= 2 and bbox_iou <= 0.01 and distance > (max_distance * 0.7):
                continue

            size_penalty = abs(1.0 - size_ratio) * 0.6
            overlap_bonus = bbox_iou * 0.45
            score = distance + size_penalty - overlap_bonus

            if best_score is None or score < best_score:
                best_score = score
                best_match_id = stable_id

        return best_match_id

    def stabilize_player_track_ids(self, tracks):
        player_frames = tracks.get("players")
        if not player_frames:
            return

        raw_to_stable_id = {}
        raw_last_seen_frame = {}
        stable_states = {}
        next_stable_id = 1

        for frame_num, frame_players in enumerate(player_frames):
            expired_raw_ids = [
                raw_track_id
                for raw_track_id, last_seen_frame in raw_last_seen_frame.items()
                if (frame_num - last_seen_frame) > self.player_id_max_gap_frames
            ]
            for raw_track_id in expired_raw_ids:
                raw_last_seen_frame.pop(raw_track_id, None)
                raw_to_stable_id.pop(raw_track_id, None)

            stabilized_frame = {}
            assigned_stable_ids = set()
            pending_players = []

            sorted_players = sorted(
                frame_players.items(),
                key=lambda item: self.normalize_track_id(item[0])
            )

            for raw_track_id, player_info in sorted_players:
                normalized_raw_track_id = self.normalize_track_id(raw_track_id)
                stable_id = raw_to_stable_id.get(normalized_raw_track_id)

                if stable_id is None or stable_id in assigned_stable_ids:
                    pending_players.append((normalized_raw_track_id, player_info))
                    continue

                raw_last_seen_frame[normalized_raw_track_id] = frame_num
                assigned_stable_ids.add(stable_id)

                stabilized_player_info = dict(player_info)
                stabilized_player_info["raw_track_id"] = normalized_raw_track_id
                stabilized_player_info["stable_track_id"] = stable_id
                stabilized_frame[stable_id] = stabilized_player_info

                self.update_stable_player_state(
                    stable_states,
                    stable_id,
                    frame_num,
                    normalized_raw_track_id,
                    stabilized_player_info
                )

            for normalized_raw_track_id, player_info in pending_players:
                stable_id = self.find_best_stable_player_match(
                    stable_states,
                    frame_num,
                    player_info,
                    assigned_stable_ids
                )

                if stable_id is None:
                    stable_id = next_stable_id
                    next_stable_id += 1

                # Keep only one live raw-track mapping for each stable player identity.
                for known_raw_track_id, known_stable_id in list(raw_to_stable_id.items()):
                    if known_stable_id == stable_id and known_raw_track_id != normalized_raw_track_id:
                        del raw_to_stable_id[known_raw_track_id]

                raw_to_stable_id[normalized_raw_track_id] = stable_id
                raw_last_seen_frame[normalized_raw_track_id] = frame_num
                assigned_stable_ids.add(stable_id)

                stabilized_player_info = dict(player_info)
                stabilized_player_info["raw_track_id"] = normalized_raw_track_id
                stabilized_player_info["stable_track_id"] = stable_id
                stabilized_frame[stable_id] = stabilized_player_info

                self.update_stable_player_state(
                    stable_states,
                    stable_id,
                    frame_num,
                    normalized_raw_track_id,
                    stabilized_player_info
                )

            tracks["players"][frame_num] = stabilized_frame

    def detect_frames(self, frames):
        batch_size = 20
        detections = []

        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.05)
            detections += detections_batch

        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            'players': [],
            'referees': [],
            'ball': []
        }

        previous_ball_center = None
        previous_ball_bbox = None
        previous_ball_velocity = np.array([0.0, 0.0], dtype=float)
        missing_ball_frames = 0

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detection_supervision.class_id[object_ind] = cls_names_inv['player']

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            # Players and referees
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # Ball selection
            ball_candidates = []
            player_bboxes = [player_info["bbox"] for player_info in tracks["players"][frame_num].values()]

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                confidence = float(frame_detection[2])
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    center = get_center_of_bbox(bbox)
                    if (
                        confidence >= self.min_ball_confidence and
                        self.is_valid_ball_candidate(bbox, frames[frame_num].shape)
                    ):
                        ball_candidates.append((bbox, confidence, center))

            predicted_ball_center = self.get_predicted_ball_center(
                previous_ball_center,
                previous_ball_velocity,
                missing_ball_frames + 1
            )

            if len(ball_candidates) > 0:
                scored_candidates = [
                    (
                        candidate,
                        self.score_ball_candidate(
                            candidate,
                            predicted_ball_center,
                            previous_ball_bbox,
                            previous_ball_velocity,
                            missing_ball_frames,
                            player_bboxes,
                            frames[frame_num].shape
                        )
                    )
                    for candidate in ball_candidates
                ]
                (best_ball_bbox, best_ball_confidence, best_ball_center), best_ball_score = max(
                    scored_candidates,
                    key=lambda item: item[1]
                )

                min_candidate_score = self.get_min_ball_candidate_score(
                    previous_ball_bbox,
                    missing_ball_frames
                )

                if best_ball_score >= min_candidate_score:
                    tracks["ball"][frame_num][1] = self.build_ball_track_info(
                        best_ball_bbox,
                        "detected",
                        confidence=best_ball_confidence,
                        score=best_ball_score
                    )

                    if previous_ball_center is not None:
                        observed_velocity = np.array([
                            best_ball_center[0] - previous_ball_center[0],
                            best_ball_center[1] - previous_ball_center[1]
                        ], dtype=float)
                        previous_ball_velocity = (0.55 * previous_ball_velocity) + (0.45 * observed_velocity)
                    else:
                        previous_ball_velocity = np.array([0.0, 0.0], dtype=float)

                    previous_ball_center = best_ball_center
                    previous_ball_bbox = best_ball_bbox
                    missing_ball_frames = 0
                elif previous_ball_bbox is not None and missing_ball_frames < self.max_ball_gap_fill:
                    predicted_bbox = self.shift_bbox(previous_ball_bbox, previous_ball_velocity)
                    if predicted_bbox is not None and self.is_valid_ball_candidate(predicted_bbox, frames[frame_num].shape):
                        tracks["ball"][frame_num][1] = self.build_ball_track_info(
                            predicted_bbox,
                            "predicted",
                            confidence=0.0,
                            score=best_ball_score
                        )
                        previous_ball_bbox = predicted_bbox
                        previous_ball_center = get_center_of_bbox(predicted_bbox)
                        previous_ball_velocity = previous_ball_velocity * 0.82
                        missing_ball_frames += 1
                    else:
                        missing_ball_frames += 1
                else:
                    missing_ball_frames += 1

                if missing_ball_frames > self.max_ball_gap_fill:
                    previous_ball_bbox = None
                    previous_ball_center = None
                    previous_ball_velocity = np.array([0.0, 0.0], dtype=float)
            elif previous_ball_bbox is not None and missing_ball_frames < self.max_ball_gap_fill:
                predicted_bbox = self.shift_bbox(previous_ball_bbox, previous_ball_velocity)
                if predicted_bbox is not None and self.is_valid_ball_candidate(predicted_bbox, frames[frame_num].shape):
                    tracks["ball"][frame_num][1] = self.build_ball_track_info(
                        predicted_bbox,
                        "predicted"
                    )
                    previous_ball_bbox = predicted_bbox
                    previous_ball_center = get_center_of_bbox(predicted_bbox)
                    previous_ball_velocity = previous_ball_velocity * 0.85
                    missing_ball_frames += 1
                else:
                    missing_ball_frames += 1
            else:
                missing_ball_frames += 1

            if missing_ball_frames > self.max_ball_gap_fill:
                previous_ball_bbox = None
                previous_ball_center = None
                previous_ball_velocity = np.array([0.0, 0.0], dtype=float)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def normalize_cv2_color(self, color):
        if color is None:
            return (0, 0, 255)

        color_array = np.array(color, dtype=float).flatten()
        if color_array.size < 3:
            return (0, 0, 255)

        clipped = np.clip(color_array[:3], 0, 255).astype(int)
        return (int(clipped[0]), int(clipped[1]), int(clipped[2]))

    def style_line_color(self, color):
        normalized = np.array(self.normalize_cv2_color(color), dtype=float)
        styled = np.clip((normalized * 0.65) + 90.0, 0, 255).astype(int)
        return (int(styled[0]), int(styled[1]), int(styled[2]))

    def draw_tactical_segment(self, overlay, start_xy, end_xy, line_color, thickness_outer, thickness_inner):
        outline_color = tuple(
            int(channel)
            for channel in np.clip((np.array(line_color, dtype=float) * 0.18) + 12.0, 0, 255)
        )
        cv2.line(overlay, start_xy, end_xy, outline_color, thickness_outer, cv2.LINE_AA)
        cv2.line(overlay, start_xy, end_xy, line_color, thickness_inner, cv2.LINE_AA)

    def draw_tactical_lines(self, frame, frame_num, team_tactical_lines=None, team_line_colors=None):
        if team_tactical_lines is None:
            return frame

        overlay = frame.copy()
        drew_anything = False

        if team_line_colors is None:
            team_line_colors = {}

        for team_id, frame_overlays in team_tactical_lines.items():
            if frame_num >= len(frame_overlays):
                continue

            tactical_overlay = frame_overlays[frame_num]
            if tactical_overlay is None:
                continue

            line_color = self.style_line_color(team_line_colors.get(team_id, (255, 255, 255)))
            for start_point, end_point in tactical_overlay.get("horizontal_segments", []):
                start_xy = (int(start_point[0]), int(start_point[1]))
                end_xy = (int(end_point[0]), int(end_point[1]))

                self.draw_tactical_segment(
                    overlay,
                    start_xy,
                    end_xy,
                    line_color,
                    thickness_outer=6,
                    thickness_inner=3
                )
                drew_anything = True

            for start_point, end_point in tactical_overlay.get("vertical_segments", []):
                start_xy = (int(start_point[0]), int(start_point[1]))
                end_xy = (int(end_point[0]), int(end_point[1]))

                self.draw_tactical_segment(
                    overlay,
                    start_xy,
                    end_xy,
                    line_color,
                    thickness_outer=4,
                    thickness_inner=2
                )
                drew_anything = True

        if drew_anything:
            cv2.addWeighted(overlay, 0.42, frame, 0.58, 0, frame)

        return frame

    def draw_ellipse(self, frame, bbox, color):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        draw_color = self.normalize_cv2_color(color)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=draw_color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        return frame
    
    def get_ball_visual_style(self, ball_info):
        source = None
        if ball_info is not None:
            source = ball_info.get("source")

        if source == "predicted":
            return {
                "arrow_color": (0, 165, 255),
                "fill_color": (0, 110, 220),
                "label": "Predicted",
                "line_style": "dotted",
            }
        if source == "interpolated":
            return {
                "arrow_color": (255, 255, 0),
                "fill_color": (255, 200, 40),
                "label": "Interpolated",
                "line_style": "dashed",
            }
        if source == "detected":
            return {
                "arrow_color": (255, 255, 255),
                "fill_color": (0, 0, 255),
                "label": "Detected",
                "line_style": "solid",
            }

        return {
            "arrow_color": (190, 190, 190),
            "fill_color": (80, 80, 80),
            "label": "Lost",
            "line_style": "solid",
        }

    def draw_styled_segment(self, frame, start_pt, end_pt, color, line_style="solid", thickness=2):
        start = np.array(start_pt, dtype=float)
        end = np.array(end_pt, dtype=float)
        distance = float(np.linalg.norm(end - start))

        if distance < 1.0:
            cv2.circle(frame, (int(start_pt[0]), int(start_pt[1])), max(1, thickness), color, -1)
            return frame

        direction = (end - start) / distance

        if line_style == "solid":
            cv2.line(frame, tuple(start.astype(int)), tuple(end.astype(int)), color, thickness, cv2.LINE_AA)
            return frame

        if line_style == "dashed":
            dash_length = 12.0
            gap_length = 7.0
            cursor = 0.0
            while cursor < distance:
                dash_end = min(cursor + dash_length, distance)
                seg_start = start + (direction * cursor)
                seg_end = start + (direction * dash_end)
                cv2.line(frame, tuple(seg_start.astype(int)), tuple(seg_end.astype(int)), color, thickness, cv2.LINE_AA)
                cursor += dash_length + gap_length
            return frame

        dot_spacing = 8.0
        dot_radius = max(1, thickness)
        cursor = 0.0
        while cursor <= distance:
            dot_center = start + (direction * cursor)
            cv2.circle(frame, tuple(dot_center.astype(int)), dot_radius, color, -1, cv2.LINE_AA)
            cursor += dot_spacing

        return frame

    def draw_ball_style_legend(self, frame, x, y):
        legend_items = [
            self.get_ball_visual_style({"source": "detected"}),
            self.get_ball_visual_style({"source": "interpolated"}),
            self.get_ball_visual_style({"source": "predicted"}),
        ]

        for index, item in enumerate(legend_items):
            item_x = x + (index * 120)
            self.draw_styled_segment(
                frame,
                (item_x, y),
                (item_x + 38, y),
                item["arrow_color"],
                item["line_style"],
                thickness=2
            )
            cv2.putText(
                frame,
                item["label"],
                (item_x + 48, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (235, 238, 242),
                1,
                cv2.LINE_AA
            )

        return frame

    def draw_mini_pitch(
        self,
        frame,
        player_dict,
        ball_dict,
        pitch_dimensions=None
    ):
        if pitch_dimensions is None:
            pitch_dimensions = (105.0, 68.0)

        pitch_length = max(1.0, float(pitch_dimensions[0]))
        pitch_width = max(1.0, float(pitch_dimensions[1]))

        frame_height, frame_width = frame.shape[:2]
        card_width = min(320, max(240, int(frame_width * 0.17)))
        card_height = int(card_width * 0.86)
        card_x1 = 24
        card_y1 = frame_height - card_height - 24
        card_x2 = card_x1 + card_width
        card_y2 = card_y1 + card_height

        card_roi = frame[card_y1:card_y2, card_x1:card_x2]
        overlay = card_roi.copy()
        overlay[:] = (22, 30, 24)
        cv2.addWeighted(overlay, 0.48, card_roi, 0.52, 0, card_roi)
        cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), (92, 110, 96), 1)

        cv2.putText(
            frame,
            "MINI PITCH",
            (card_x1 + 14, card_y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (234, 239, 236),
            1,
            cv2.LINE_AA
        )

        padding_x = 18
        padding_top = 34
        padding_bottom = 16
        pitch_x1 = card_x1 + padding_x
        pitch_y1 = card_y1 + padding_top
        pitch_x2 = card_x2 - padding_x
        pitch_y2 = card_y2 - padding_bottom
        pitch_w = pitch_x2 - pitch_x1
        pitch_h = pitch_y2 - pitch_y1

        cv2.rectangle(frame, (pitch_x1, pitch_y1), (pitch_x2, pitch_y2), (178, 220, 182), 1, cv2.LINE_AA)
        cv2.line(
            frame,
            (pitch_x1 + (pitch_w // 2), pitch_y1),
            (pitch_x1 + (pitch_w // 2), pitch_y2),
            (108, 155, 114),
            1,
            cv2.LINE_AA
        )
        center_point = (pitch_x1 + (pitch_w // 2), pitch_y1 + (pitch_h // 2))
        center_radius = max(8, int(min(pitch_w, pitch_h) * 0.12))
        cv2.circle(frame, center_point, center_radius, (108, 155, 114), 1, cv2.LINE_AA)

        box_depth = max(10, int(pitch_w * 0.12))
        box_half_height = max(18, int(pitch_h * 0.22))
        cv2.rectangle(
            frame,
            (pitch_x1, center_point[1] - box_half_height),
            (pitch_x1 + box_depth, center_point[1] + box_half_height),
            (108, 155, 114),
            1,
            cv2.LINE_AA
        )
        cv2.rectangle(
            frame,
            (pitch_x2 - box_depth, center_point[1] - box_half_height),
            (pitch_x2, center_point[1] + box_half_height),
            (108, 155, 114),
            1,
            cv2.LINE_AA
        )

        def to_mini_pitch(point):
            if point is None or len(point) < 2:
                return None

            norm_x = float(point[0]) / pitch_length
            norm_y = float(point[1]) / pitch_width
            norm_x = min(max(norm_x, 0.0), 1.0)
            norm_y = min(max(norm_y, 0.0), 1.0)

            draw_x = int(pitch_x1 + (norm_x * pitch_w))
            draw_y = int(pitch_y1 + (norm_y * pitch_h))
            return (draw_x, draw_y)

        for player in player_dict.values():
            transformed_position = player.get("position_transformed")
            draw_point = to_mini_pitch(transformed_position)
            if draw_point is None:
                continue

            team_color = self.normalize_cv2_color(player.get("team_color", (180, 180, 180)))
            radius = 5 if player.get("has_ball") else 4
            cv2.circle(frame, draw_point, radius + 1, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, draw_point, radius, team_color, -1, cv2.LINE_AA)

        ball_info = ball_dict.get(1)
        if ball_info is not None:
            ball_position = ball_info.get("position_transformed")
            draw_ball = to_mini_pitch(ball_position)
            if draw_ball is not None:
                ball_style = self.get_ball_visual_style(ball_info)
                cv2.circle(frame, draw_ball, 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, draw_ball, 3, ball_style["fill_color"], -1, cv2.LINE_AA)

        return frame

    def draw_ball_pointer(self, frame, ball_center, color=(255, 255, 255), fill_color=(0, 0, 255)):
        if ball_center is None:
            return frame

        x, y = ball_center

        # arrow start slightly above-right of ball
        start_pt = (x + 35, y - 35)
        end_pt = (x, y)

        cv2.arrowedLine(
            frame,
            start_pt,
            end_pt,
            color,
            3,
            tipLength=0.35
        )

        cv2.circle(frame, (x, y), 6, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 3, fill_color, -1)

        return frame

    def draw_arrow_trail(self, frame, ball_history):
        valid_points = [entry for entry in ball_history if entry is not None and entry.get("point") is not None]

        if len(valid_points) == 0:
            return frame

        if len(valid_points) == 1:
            style = self.get_ball_visual_style(valid_points[-1].get("ball_info"))
            cv2.circle(frame, valid_points[-1]["point"], 5, (255, 255, 255), -1)
            cv2.circle(frame, valid_points[-1]["point"], 3, style["fill_color"], -1)
            return frame

        valid_points = valid_points[-5:]

        for i in range(1, len(valid_points) - 1):
            style = self.get_ball_visual_style(valid_points[i].get("ball_info"))
            self.draw_styled_segment(
                frame,
                valid_points[i - 1]["point"],
                valid_points[i]["point"],
                style["arrow_color"],
                style["line_style"],
                thickness=2
            )

        start_pt = valid_points[-2]["point"]
        end_pt = valid_points[-1]["point"]
        end_style = self.get_ball_visual_style(valid_points[-1].get("ball_info"))

        # Only draw arrow if there is real movement
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        dist = (dx * dx + dy * dy) ** 0.5

        if dist >= 3:
            self.draw_styled_segment(
                frame,
                start_pt,
                end_pt,
                end_style["arrow_color"],
                end_style["line_style"],
                thickness=3
            )
            arrow_tail = (
                int(start_pt[0] + (0.72 * dx)),
                int(start_pt[1] + (0.72 * dy))
            )
            cv2.arrowedLine(
                frame,
                arrow_tail,
                end_pt,
                end_style["arrow_color"],
                3,
                tipLength=0.35
            )
        else:
            self.draw_styled_segment(
                frame,
                start_pt,
                end_pt,
                end_style["arrow_color"],
                end_style["line_style"],
                thickness=2
            )

        cv2.circle(frame, end_pt, 5, (255, 255, 255), -1)
        cv2.circle(frame, end_pt, 3, end_style["fill_color"], -1)

        return frame

    def draw_ball_carrier_indicator(self, frame, bbox, color):
        x_center, y_center = get_foot_position(bbox)
        indicator_color = self.normalize_cv2_color(color)

        cv2.circle(frame, (int(x_center), int(y_center - 12)), 18, (255, 255, 255), 3)
        cv2.circle(frame, (int(x_center), int(y_center - 12)), 14, indicator_color, 2)
        cv2.putText(
            frame,
            "BALL",
            (int(x_center - 22), int(y_center - 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2
        )

        return frame

    def format_match_clock(self, frame_num, video_fps):
        fps = max(float(video_fps or 0.0), 1e-6)
        elapsed_seconds = int(frame_num / fps)
        minutes = elapsed_seconds // 60
        seconds = elapsed_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"
    
    def draw_scoreboard(
        self,
        frame,
        frame_num,
        team_ball_control,
        team_touch_count_per_frame,
        team_pass_count_per_frame,
        threat_info,
        team_formations=None,
        team_shape_per_frame=None,
        team_shape_summaries=None,
        ball_info=None,
        pass_event=None,
        video_fps=24.0
    ):
        frame_height, frame_width = frame.shape[:2]

        panel_width = min(520, max(420, int(frame_width * 0.27)))
        panel_height = 390
        x1 = frame_width - panel_width - 24
        y1 = 24
        x2 = x1 + panel_width
        y2 = y1 + panel_height

        panel_roi = frame[y1:y2, x1:x2]
        overlay = panel_roi.copy()
        overlay[:] = (24, 28, 34)
        alpha = 0.52
        cv2.addWeighted(overlay, alpha, panel_roi, 1 - alpha, 0, panel_roi)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 110, 130), 1)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        team_1_num_frames = sum(1 for t in team_ball_control_till_frame if t == 1)
        team_2_num_frames = sum(1 for t in team_ball_control_till_frame if t == 2)

        total_frames = team_1_num_frames + team_2_num_frames
        if total_frames == 0:
            team_1_possession = 0
            team_2_possession = 0
        else:
            team_1_possession = (team_1_num_frames / total_frames) * 100
            team_2_possession = (team_2_num_frames / total_frames) * 100

        threat_label = "No Threat"
        threat_score = 0
        threat_progress_pct = 0
        if threat_info is not None:
            threat_label = threat_info.get("label", "No Threat")
            threat_score = threat_info.get("score", 0)
            threat_progress_pct = int(round(100 * threat_info.get("progress_ratio", 0.0)))

        if frame_num < len(team_touch_count_per_frame):
            current_touch_counts = team_touch_count_per_frame[frame_num]
        else:
            current_touch_counts = {1: 0, 2: 0}

        if team_pass_count_per_frame is not None and frame_num < len(team_pass_count_per_frame):
            current_pass_counts = team_pass_count_per_frame[frame_num]
        else:
            current_pass_counts = {1: 0, 2: 0}

        accent_color = (147, 186, 214)
        primary_text_color = (240, 244, 248)
        secondary_text_color = (192, 202, 214)
        detail_text_color = (213, 222, 232)
        clock_text = self.format_match_clock(frame_num, video_fps)
        live_control_team = int(team_ball_control[frame_num]) if frame_num < len(team_ball_control) else 0
        control_label = f"T{live_control_team}" if live_control_team in (1, 2) else "--"
        control_color = secondary_text_color
        if live_control_team == 1:
            control_color = (82, 199, 122)
        elif live_control_team == 2:
            control_color = (108, 171, 255)

        cv2.putText(
            frame,
            "SOCCER ANALYTICS",
            (x1 + 18, y1 + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            primary_text_color,
            2,
            cv2.LINE_AA
        )
        cv2.line(frame, (x1 + 18, y1 + 46), (x2 - 18, y1 + 46), accent_color, 1, cv2.LINE_AA)

        if team_formations is None:
            team_formations = {}

        if team_shape_per_frame is None:
            team_shape_per_frame = {}

        if team_shape_summaries is None:
            team_shape_summaries = {}

        team_1_formation = team_formations.get(1, "Unknown")
        team_2_formation = team_formations.get(2, "Unknown")
        team_1_shape_summary = team_shape_summaries.get(1, {})
        team_2_shape_summary = team_shape_summaries.get(2, {})

        team_1_shape_label = team_1_shape_summary.get("shape_label", "Unknown")
        team_2_shape_label = team_2_shape_summary.get("shape_label", "Unknown")

        team_1_shape_metrics = team_1_shape_summary
        team_2_shape_metrics = team_2_shape_summary

        if 1 in team_shape_per_frame and frame_num < len(team_shape_per_frame[1]):
            if team_shape_per_frame[1][frame_num] is not None:
                team_1_shape_metrics = team_shape_per_frame[1][frame_num]

        if 2 in team_shape_per_frame and frame_num < len(team_shape_per_frame[2]):
            if team_shape_per_frame[2][frame_num] is not None:
                team_2_shape_metrics = team_shape_per_frame[2][frame_num]

        def format_shape_line(shape_metrics):
            if shape_metrics is None or "width_pct" not in shape_metrics:
                return "W:-- L:-- C:-- DL:--"

            return (
                f"W:{shape_metrics['width_pct']:.0f}% "
                f"L:{shape_metrics['length_pct']:.0f}% "
                f"C:{shape_metrics['compactness_pct']:.0f}% "
                f"DL:{shape_metrics['line_height_pct']:.0f}%"
            )

        cv2.putText(
            frame,
            f"Possession  T1 {team_1_possession:.1f}% | T2 {team_2_possession:.1f}%",
            (x1 + 18, y1 + 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            primary_text_color,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Touches  {current_touch_counts.get(1, 0)} / {current_touch_counts.get(2, 0)}    "
            f"Passes  {current_pass_counts.get(1, 0)} / {current_pass_counts.get(2, 0)}",
            (x1 + 18, y1 + 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            primary_text_color,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Threat  {threat_label} ({threat_score}) | Prog {threat_progress_pct}%",
            (x1 + 18, y1 + 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            primary_text_color,
            1,
            cv2.LINE_AA
        )

        ball_style = self.get_ball_visual_style(ball_info)
        ball_status_text = f"Ball Track: {ball_style['label']}"
        if ball_info is not None and ball_info.get("confidence", 0) > 0:
            ball_status_text += f" ({ball_info['confidence']:.2f})"

        cv2.putText(
            frame,
            ball_status_text,
            (x1 + 18, y1 + 168),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            ball_style["fill_color"],
            1,
            cv2.LINE_AA
        )
        self.draw_ball_style_legend(frame, x1 + 18, y1 + 196)

        cv2.putText(
            frame,
            f"Clock  {clock_text}",
            (x1 + 18, y1 + 226),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            primary_text_color,
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            f"Control  {control_label}",
            (x1 + 160, y1 + 226),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            control_color,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"T1  {team_1_formation} | {team_1_shape_label}",
            (x1 + 18, y1 + 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            primary_text_color,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            format_shape_line(team_1_shape_metrics),
            (x1 + 18, y1 + 278),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            detail_text_color,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"T2  {team_2_formation} | {team_2_shape_label}",
            (x1 + 18, y1 + 315),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            primary_text_color,
            1,
            cv2.LINE_AA
        )

        ball_style = self.get_ball_visual_style(ball_info)
        cv2.putText(
            frame,
            format_shape_line(team_2_shape_metrics),
            (x1 + 18, y1 + 343),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            detail_text_color,
            1,
            cv2.LINE_AA
        )

        if pass_event is None:
            pass_event_text = "Last Pass: --"
            pass_event_color = secondary_text_color
        else:
            pass_event_text = (
                f"Last Pass: T{pass_event.get('team_id', '?')} "
                f"P{pass_event.get('from_player', '?')} -> P{pass_event.get('to_player', '?')}"
            )
            pass_event_color = (24, 110, 54)

        cv2.putText(
            frame,
            pass_event_text,
            (x1 + 18, y1 + 376),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            pass_event_color,
            1,
            cv2.LINE_AA
        )

        return frame

    def draw_annotations(
        self,
        video_frames,
        tracks,
        team_ball_control,
        team_touch_count_per_frame,
        team_pass_count_per_frame=None,
        team_threat_per_frame=None,
        team_formations=None,
        team_shape_per_frame=None,
        team_shape_summaries=None,
        team_tactical_lines=None,
        team_line_colors=None,
        pass_events_per_frame=None,
        pitch_dimensions=None,
        video_fps=24.0,
        copy_frames=True
    ):
        ball_history = []

        total_frames = min(
            len(video_frames),
            len(tracks["players"]),
            len(tracks["ball"]),
            len(tracks["referees"]),
            len(team_ball_control)
        )

        if copy_frames:
            output_video_frames = []
        else:
            output_video_frames = video_frames[:total_frames]

        for frame_num in range(total_frames):
            if copy_frames:
                frame = video_frames[frame_num].copy()
            else:
                frame = output_video_frames[frame_num]

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            frame = self.draw_mini_pitch(
                frame,
                player_dict,
                ball_dict,
                pitch_dimensions=pitch_dimensions
            )

            frame = self.draw_tactical_lines(
                frame,
                frame_num,
                team_tactical_lines,
                team_line_colors
            )

            for _, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                if player.get("has_ball"):
                    frame = self.draw_ball_carrier_indicator(frame, player["bbox"], color)
                frame = self.draw_ellipse(frame, player["bbox"], color)

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            current_ball_center = None
            current_ball_info = None
            for _, ball in ball_dict.items():
                current_ball_info = ball
                current_ball_center = get_center_of_bbox(ball["bbox"])
                break

            ball_history.append({
                "point": current_ball_center,
                "ball_info": current_ball_info,
            })
            ball_history = ball_history[-8:]
            frame = self.draw_arrow_trail(frame, ball_history)
            ball_style = self.get_ball_visual_style(current_ball_info)
            frame = self.draw_ball_pointer(
                frame,
                current_ball_center,
                color=ball_style["arrow_color"],
                fill_color=ball_style["fill_color"]
            )

            threat_info = None
            if team_threat_per_frame is not None and frame_num < len(team_threat_per_frame):
                threat_info = team_threat_per_frame[frame_num]

            frame = self.draw_scoreboard(
                frame,
                frame_num,
                team_ball_control,
                team_touch_count_per_frame,
                team_pass_count_per_frame,
                threat_info,
                team_formations,
                team_shape_per_frame,
                team_shape_summaries,
                current_ball_info,
                pass_events_per_frame[frame_num] if pass_events_per_frame is not None and frame_num < len(pass_events_per_frame) else None,
                video_fps
            )

            if copy_frames:
                output_video_frames.append(frame)

        return output_video_frames
