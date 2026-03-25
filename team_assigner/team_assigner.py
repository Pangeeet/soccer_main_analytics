from sklearn.cluster import KMeans
import numpy as np
import cv2


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.player_team_votes = {}
        self.kmeans = None
        self.vote_window = 25

    def get_clustering_model(self, pixels):
        if pixels is None or len(pixels) < 2:
            return None

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=5, random_state=0)
        kmeans.fit(pixels)

        return kmeans

    def extract_torso_pixels(self, frame, bbox):
        height, width, _ = frame.shape

        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            return None

        bbox_width = x2 - x1
        bbox_height = y2 - y1
        if bbox_width < 8 or bbox_height < 15:
            return None

        image = frame[y1:y2, x1:x2]
        if image.size == 0:
            return None

        torso_y1 = int(image.shape[0] * 0.15)
        torso_y2 = int(image.shape[0] * 0.55)
        torso_x1 = int(image.shape[1] * 0.2)
        torso_x2 = int(image.shape[1] * 0.8)
        torso_image = image[torso_y1:torso_y2, torso_x1:torso_x2]

        if torso_image.size == 0:
            return None

        hsv_image = cv2.cvtColor(torso_image, cv2.COLOR_BGR2HSV)
        green_mask = (
            (hsv_image[:, :, 0] >= 35) &
            (hsv_image[:, :, 0] <= 95) &
            (hsv_image[:, :, 1] >= 40) &
            (hsv_image[:, :, 2] >= 40)
        )

        valid_pixels = torso_image[~green_mask]
        if len(valid_pixels) < 20:
            valid_pixels = torso_image.reshape(-1, 3)

        if len(valid_pixels) == 0:
            return None

        return valid_pixels.reshape(-1, 3)

    def get_player_color(self, frame, bbox):
        pixels = self.extract_torso_pixels(frame, bbox)
        if pixels is None or len(pixels) < 2:
            return np.array([0, 0, 0], dtype=np.float32)

        kmeans = self.get_clustering_model(pixels)
        if kmeans is None:
            return np.array([0, 0, 0], dtype=np.float32)

        labels = kmeans.labels_
        label_counts = np.bincount(labels, minlength=2)
        dominant_cluster = int(np.argmax(label_counts))
        player_color = kmeans.cluster_centers_[dominant_cluster].astype(np.float32)

        return player_color

    def predict_team_from_color(self, player_color):
        if len(self.team_colors) < 2:
            return 1

        distance_team_1 = np.linalg.norm(player_color - self.team_colors[1])
        distance_team_2 = np.linalg.norm(player_color - self.team_colors[2])
        return 1 if distance_team_1 <= distance_team_2 else 2

    def assign_team_color(self, video_frames, player_tracks, sample_frames=60):
        player_colors = []
        available_frames = min(len(video_frames), len(player_tracks))
        total_frames = min(sample_frames, available_frames)
        if total_frames <= 0:
            self.team_colors[1] = np.array([255, 0, 0])
            self.team_colors[2] = np.array([0, 0, 255])
            self.kmeans = None
            return

        sampled_frame_indices = sorted({
            int(frame_num)
            for frame_num in np.linspace(0, available_frames - 1, num=total_frames)
        })

        for frame_num in sampled_frame_indices:
            frame = video_frames[frame_num]
            player_detections = player_tracks[frame_num]

            for _, player_detection in player_detections.items():
                bbox = player_detection["bbox"]
                player_color = self.get_player_color(frame, bbox)
                if not np.array_equal(player_color, np.array([0, 0, 0], dtype=np.float32)):
                    player_colors.append(player_color)

        if len(player_colors) < 4:
            self.team_colors[1] = np.array([255, 0, 0])
            self.team_colors[2] = np.array([0, 0, 255])
            self.kmeans = None
            return

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0].astype(np.float32)
        self.team_colors[2] = kmeans.cluster_centers_[1].astype(np.float32)

    def get_player_team(self, frame, player_bbox, player_id):
        player_color = self.get_player_color(frame, player_bbox)

        if np.array_equal(player_color, np.array([0, 0, 0], dtype=np.float32)):
            return self.player_team_dict.get(player_id, 1)

        predicted_team = self.predict_team_from_color(player_color)
        votes = self.player_team_votes.setdefault(player_id, [])
        votes.append(predicted_team)
        if len(votes) > self.vote_window:
            votes.pop(0)

        team_1_votes = votes.count(1)
        team_2_votes = votes.count(2)
        team_id = 1 if team_1_votes >= team_2_votes else 2

        self.player_team_dict[player_id] = team_id

        return team_id
