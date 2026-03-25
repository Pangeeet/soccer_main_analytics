import sys
sys.path.append('../')
from utils import get_center_of_bbox, measure_distance, get_foot_position

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 95
        self.last_assigned_player = -1

    def is_ball_close_enough(self, player_bbox, ball_position):
        x1, y1, x2, y2 = player_bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)

        ball_x, ball_y = ball_position
        horizontal_margin = width * 0.45
        horizontal_ok = (x1 - horizontal_margin) <= ball_x <= (x2 + horizontal_margin)
        vertical_ok = (y1 + height * 0.18) <= ball_y <= (y2 + height * 0.32)

        foot_position = get_foot_position(player_bbox)
        foot_distance = measure_distance(foot_position, ball_position)
        dynamic_max_distance = min(self.max_player_ball_distance, max(30, height * 0.85))

        return horizontal_ok and vertical_ok and foot_distance <= dynamic_max_distance

    def assign_ball_to_player(self, players, ball_bbox):
        if ball_bbox is None:
            self.last_assigned_player = -1
            return -1

        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            if player.get("position_transformed") is None:
                continue

            player_bbox = player['bbox']
            if not self.is_ball_close_enough(player_bbox, ball_position):
                continue

            foot_position = get_foot_position(player_bbox)
            body_center = ((player_bbox[0] + player_bbox[2]) / 2, (player_bbox[1] + player_bbox[3]) / 2)
            foot_distance = measure_distance(foot_position, ball_position)
            center_distance = measure_distance(body_center, ball_position)
            distance = foot_distance + (0.25 * center_distance)

            if player_id == self.last_assigned_player:
                distance -= 10

            if distance < minimum_distance:
                minimum_distance = distance
                assigned_player = player_id

        self.last_assigned_player = assigned_player
        return assigned_player
