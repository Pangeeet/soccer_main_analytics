import numpy as np


class FormationAnalyzer:
    def __init__(
        self,
        row_threshold=10.0,
        min_frames_per_player=10,
        pitch_length=105.0,
        pitch_width=68.0,
        max_vertical_connectors=3,
        horizontal_gap_limit_ratio=0.36,
        vertical_lane_tolerance_ratio=0.24
    ):
        self.row_threshold = row_threshold
        self.min_frames_per_player = min_frames_per_player
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.max_vertical_connectors = max_vertical_connectors
        self.horizontal_gap_limit_ratio = horizontal_gap_limit_ratio
        self.vertical_lane_tolerance_ratio = vertical_lane_tolerance_ratio
        self.formation_templates = [
            (4, 4, 2),
            (4, 3, 3),
            (4, 5, 1),
            (3, 5, 2),
            (3, 4, 3),
            (5, 3, 2),
            (5, 4, 1),
            (4, 2, 3, 1),
            (4, 1, 4, 1),
            (4, 3, 2, 1),
            (3, 4, 2, 1),
            (3, 4, 1, 2),
        ]

    def collect_average_positions(self, tracks, team_id):
        player_positions = {}

        for frame_players in tracks["players"]:
            for player_id, player in frame_players.items():
                if player.get("team") != team_id:
                    continue

                pos = player.get("position_transformed")
                if pos is None:
                    continue

                if player_id not in player_positions:
                    player_positions[player_id] = []

                player_positions[player_id].append(pos)

        avg_positions = []
        for player_id, positions in player_positions.items():
            frame_count = len(positions)
            if frame_count < self.min_frames_per_player:
                continue

            avg_x = float(np.median([p[0] for p in positions]))
            avg_y = float(np.median([p[1] for p in positions]))
            avg_positions.append((player_id, avg_x, avg_y, frame_count))

        return avg_positions

    def directional_x(self, x_value, defending_side):
        if defending_side == "right":
            return self.pitch_length - x_value
        return x_value

    def identify_goalkeeper_and_order(self, avg_positions):
        if len(avg_positions) < 2:
            return None, [], None, "Not enough players to identify goalkeeper."

        avg_positions = sorted(avg_positions, key=lambda x: x[1])
        left_gap = avg_positions[1][1] - avg_positions[0][1]
        right_gap = avg_positions[-1][1] - avg_positions[-2][1]

        if right_gap > left_gap:
            goalkeeper = avg_positions[-1]
            defending_side = "right"
            outfield_players = sorted(
                avg_positions[:-1],
                key=lambda player: self.directional_x(player[1], defending_side)
            )
        else:
            goalkeeper = avg_positions[0]
            defending_side = "left"
            outfield_players = sorted(
                avg_positions[1:],
                key=lambda player: self.directional_x(player[1], defending_side)
            )

        return goalkeeper, outfield_players, defending_side, None

    def generate_partitions(self, player_count, num_lines):
        partitions = []

        def backtrack(remaining_players, remaining_lines, current_partition):
            if remaining_lines == 0:
                if remaining_players == 0:
                    partitions.append(tuple(current_partition))
                return

            min_size = 1
            max_size = min(5, remaining_players - (remaining_lines - 1))
            for line_size in range(min_size, max_size + 1):
                current_partition.append(line_size)
                backtrack(remaining_players - line_size, remaining_lines - 1, current_partition)
                current_partition.pop()

        backtrack(player_count, num_lines, [])
        return partitions

    def score_partition(self, x_values, partition):
        score = 0.0
        line_means = []
        start_idx = 0

        for line_size in partition:
            line_x = x_values[start_idx:start_idx + line_size]
            if len(line_x) == 0:
                return float("inf")

            score += float(np.var(line_x)) * (1.0 + (0.2 if line_size == 1 else 0.0))
            line_means.append(float(np.mean(line_x)))
            start_idx += line_size

        for idx in range(len(line_means) - 1):
            gap = line_means[idx + 1] - line_means[idx]
            if gap <= 0:
                return float("inf")
            score -= gap * 0.35

        return score

    def choose_best_partition_from_x_values(self, x_values):
        if len(x_values) < 6:
            return []

        candidate_partitions = self.get_candidate_partitions(len(x_values))
        if not candidate_partitions:
            return []

        best_partition = []
        best_score = float("inf")

        for partition in candidate_partitions:
            score = self.score_partition(x_values, partition)
            if score < best_score:
                best_score = score
                best_partition = list(partition)

        return best_partition

    def get_candidate_partitions(self, player_count):
        if player_count <= 0:
            return []

        candidate_partitions = []
        seen_partitions = set()

        for template in self.formation_templates:
            if player_count == sum(template):
                candidate = tuple(template)
            else:
                candidate = tuple(self.scale_line_counts(list(template), player_count))

            if (
                len(candidate) < 2 or
                sum(candidate) != player_count or
                candidate in seen_partitions
            ):
                continue

            seen_partitions.add(candidate)
            candidate_partitions.append(list(candidate))

        return candidate_partitions

    def choose_best_partition(self, outfield_players, defending_side):
        if len(outfield_players) < 8:
            return []

        x_values = [self.directional_x(player[1], defending_side) for player in outfield_players]
        return self.choose_best_partition_from_x_values(x_values)

    def normalize_line_counts(self, line_counts):
        total = sum(line_counts)
        if total <= 0:
            return []

        candidates = self.get_candidate_partitions(total)
        if not candidates:
            return list(line_counts)

        best_candidate = min(
            candidates,
            key=lambda candidate: (
                0 if tuple(candidate) == tuple(line_counts) else 1,
                len(candidate) != len(line_counts),
                sum(abs(a - b) for a, b in zip(candidate, line_counts[:len(candidate)])),
                candidate
            )
        )
        return list(best_candidate)

    def compute_frame_shape_metrics(self, frame_players, team_id, goalkeeper_id, stable_player_ids, defending_side):
        stable_outfield_positions = []
        fallback_outfield_positions = []

        for player_id, player in frame_players.items():
            if player.get("team") != team_id:
                continue

            position = player.get("position_transformed")
            if position is None or player_id == goalkeeper_id:
                continue

            fallback_outfield_positions.append(position)
            if not stable_player_ids or player_id in stable_player_ids:
                stable_outfield_positions.append(position)

        positions = stable_outfield_positions if len(stable_outfield_positions) >= 4 else fallback_outfield_positions
        if len(positions) < 4:
            return None

        x_values = np.array([position[0] for position in positions], dtype=float)
        y_values = np.array([position[1] for position in positions], dtype=float)

        x_low = float(np.percentile(x_values, 10))
        x_high = float(np.percentile(x_values, 90))
        y_low = float(np.percentile(y_values, 10))
        y_high = float(np.percentile(y_values, 90))

        width = max(0.0, y_high - y_low)
        length = max(0.0, x_high - x_low)

        if defending_side == "right":
            defensive_line_x = float(np.percentile(x_values, 85))
            line_height = max(0.0, self.pitch_length - defensive_line_x)
            centroid_progress = max(0.0, self.pitch_length - float(np.mean(x_values)))
        else:
            defensive_line_x = float(np.percentile(x_values, 15))
            line_height = max(0.0, defensive_line_x)
            centroid_progress = max(0.0, float(np.mean(x_values)))

        width_pct = (width / self.pitch_width) * 100 if self.pitch_width > 0 else 0.0
        length_pct = (length / self.pitch_length) * 100 if self.pitch_length > 0 else 0.0
        line_height_pct = (line_height / self.pitch_length) * 100 if self.pitch_length > 0 else 0.0
        centroid_progress_pct = (centroid_progress / self.pitch_length) * 100 if self.pitch_length > 0 else 0.0
        compactness_pct = (
            (width * length) / (self.pitch_width * self.pitch_length) * 100
            if self.pitch_width > 0 and self.pitch_length > 0 else 0.0
        )

        return {
            "player_count": len(positions),
            "width_pct": width_pct,
            "length_pct": length_pct,
            "line_height_pct": line_height_pct,
            "centroid_progress_pct": centroid_progress_pct,
            "compactness_pct": compactness_pct,
        }

    def classify_shape(self, avg_line_height_pct, avg_compactness_pct):
        if avg_line_height_pct < 30:
            block_label = "Low block"
        elif avg_line_height_pct < 55:
            block_label = "Mid block"
        else:
            block_label = "High line"

        if avg_compactness_pct < 16:
            compactness_label = "compact"
        elif avg_compactness_pct < 28:
            compactness_label = "balanced"
        else:
            compactness_label = "stretched"

        return f"{block_label}, {compactness_label}"

    def analyze_team_shape(self, tracks, team_id, formation_details):
        goalkeeper_id = formation_details.get("goalkeeper_id")
        defending_side = formation_details.get("defending_side", "left")
        stable_player_ids = set(formation_details.get("stable_player_ids", []))

        frame_metrics = []
        for frame_players in tracks["players"]:
            frame_metrics.append(
                self.compute_frame_shape_metrics(
                    frame_players,
                    team_id,
                    goalkeeper_id,
                    stable_player_ids,
                    defending_side
                )
            )

        valid_metrics = [metric for metric in frame_metrics if metric is not None]
        if not valid_metrics:
            return {
                "shape_label": "Unknown",
                "defending_side": defending_side,
                "reason": "Not enough valid team-shape frames."
            }, frame_metrics

        avg_width_pct = float(np.mean([metric["width_pct"] for metric in valid_metrics]))
        avg_length_pct = float(np.mean([metric["length_pct"] for metric in valid_metrics]))
        avg_line_height_pct = float(np.mean([metric["line_height_pct"] for metric in valid_metrics]))
        avg_centroid_progress_pct = float(np.mean([metric["centroid_progress_pct"] for metric in valid_metrics]))
        avg_compactness_pct = float(np.mean([metric["compactness_pct"] for metric in valid_metrics]))
        avg_player_count = float(np.mean([metric["player_count"] for metric in valid_metrics]))

        return {
            "shape_label": self.classify_shape(avg_line_height_pct, avg_compactness_pct),
            "defending_side": defending_side,
            "avg_width_pct": avg_width_pct,
            "avg_length_pct": avg_length_pct,
            "avg_line_height_pct": avg_line_height_pct,
            "avg_centroid_progress_pct": avg_centroid_progress_pct,
            "avg_compactness_pct": avg_compactness_pct,
            "avg_player_count": avg_player_count,
            "valid_frames": len(valid_metrics),
        }, frame_metrics

    def scale_line_counts(self, base_counts, player_count):
        if not base_counts or player_count <= 0:
            return []

        if player_count <= len(base_counts):
            return [1] * player_count

        weights = np.array(base_counts, dtype=float)
        weights = weights / np.sum(weights)
        scaled_counts = np.maximum(1, np.floor(weights * player_count).astype(int))

        while int(np.sum(scaled_counts)) < player_count:
            idx = int(np.argmax(weights - (scaled_counts / max(player_count, 1))))
            scaled_counts[idx] += 1

        while int(np.sum(scaled_counts)) > player_count:
            candidate_indices = [idx for idx, count in enumerate(scaled_counts) if count > 1]
            if not candidate_indices:
                break
            idx = max(candidate_indices, key=lambda candidate_idx: scaled_counts[candidate_idx])
            scaled_counts[idx] -= 1

        return scaled_counts.astype(int).tolist()

    def collect_frame_line_groups(self, frame_players, team_id, formation_details):
        goalkeeper_id = formation_details.get("goalkeeper_id")
        defending_side = formation_details.get("defending_side", "left")
        stable_player_ids = set(formation_details.get("stable_player_ids", []))

        preferred_players = []
        fallback_players = []

        for player_id, player in frame_players.items():
            if player.get("team") != team_id or player_id == goalkeeper_id:
                continue

            transformed_position = player.get("position_transformed")
            screen_position = player.get("position")
            if transformed_position is None or screen_position is None:
                continue

            player_info = {
                "player_id": player_id,
                "directional_x": float(self.directional_x(transformed_position[0], defending_side)),
                "transformed_y": float(transformed_position[1]),
                "screen_position": (int(screen_position[0]), int(screen_position[1])),
            }

            fallback_players.append(player_info)
            if not stable_player_ids or player_id in stable_player_ids:
                preferred_players.append(player_info)

        working_players = fallback_players
        if len(fallback_players) > 10 and len(preferred_players) >= 4:
            working_players = preferred_players
        if len(working_players) < 4:
            return []

        working_players = sorted(working_players, key=lambda player: player["directional_x"])

        partition = self.choose_best_partition_from_x_values(
            [player["directional_x"] for player in working_players]
        )
        if not partition:
            partition = self.scale_line_counts(
                formation_details.get("line_counts", []),
                len(working_players)
            )

        if not partition:
            return []

        line_groups = []
        start_idx = 0
        for line_size in partition:
            end_idx = min(len(working_players), start_idx + line_size)
            if end_idx <= start_idx:
                break

            group = sorted(
                working_players[start_idx:end_idx],
                key=lambda player: player["transformed_y"]
            )
            line_groups.append(group)
            start_idx = end_idx

        if start_idx < len(working_players):
            leftovers = sorted(
                working_players[start_idx:],
                key=lambda player: player["transformed_y"]
            )
            if line_groups:
                line_groups[-1].extend(leftovers)
                line_groups[-1] = sorted(line_groups[-1], key=lambda player: player["transformed_y"])
            else:
                line_groups.append(leftovers)

        return line_groups

    def get_anchor_indices(self, player_count, connector_count):
        if player_count <= 0 or connector_count <= 0:
            return []

        if connector_count >= player_count:
            return list(range(player_count))

        anchor_positions = np.linspace(0, player_count - 1, connector_count)
        anchor_indices = sorted({int(round(position)) for position in anchor_positions})

        while len(anchor_indices) < connector_count:
            for candidate_idx in range(player_count):
                if candidate_idx not in anchor_indices:
                    anchor_indices.append(candidate_idx)
                if len(anchor_indices) == connector_count:
                    break

        return sorted(anchor_indices)

    def collect_horizontal_pairs(self, line_group):
        if len(line_group) < 2:
            return []

        lateral_gaps = [
            abs(line_group[idx + 1]["transformed_y"] - line_group[idx]["transformed_y"])
            for idx in range(len(line_group) - 1)
        ]
        if not lateral_gaps:
            return []

        typical_gap = float(np.median(lateral_gaps))
        max_gap = max(
            6.0,
            self.pitch_width * self.horizontal_gap_limit_ratio,
            typical_gap * 1.55
        )

        horizontal_pairs = []
        for idx in range(len(line_group) - 1):
            if lateral_gaps[idx] <= max_gap:
                horizontal_pairs.append((line_group[idx], line_group[idx + 1]))

        return horizontal_pairs

    def collect_vertical_pairs(self, current_line, next_line):
        if len(current_line) == 0 or len(next_line) == 0:
            return []

        connector_count = min(
            self.max_vertical_connectors,
            len(current_line),
            len(next_line)
        )
        if connector_count <= 0:
            return []

        current_indices = self.get_anchor_indices(len(current_line), connector_count)
        next_indices = self.get_anchor_indices(len(next_line), connector_count)
        lane_tolerance = max(5.0, self.pitch_width * self.vertical_lane_tolerance_ratio)

        vertical_pairs = []
        for current_idx, next_idx in zip(current_indices, next_indices):
            current_player = current_line[current_idx]
            next_player = next_line[next_idx]

            lateral_gap = abs(
                current_player["transformed_y"] - next_player["transformed_y"]
            )
            if lateral_gap <= lane_tolerance:
                vertical_pairs.append((current_player, next_player))

        if len(vertical_pairs) == 0:
            current_center = current_line[len(current_line) // 2]
            next_center = next_line[len(next_line) // 2]
            vertical_pairs.append((current_center, next_center))

        return vertical_pairs

    def build_frame_tactical_overlay(self, frame_players, team_id, formation_details):
        line_groups = self.collect_frame_line_groups(frame_players, team_id, formation_details)
        if len(line_groups) < 2:
            return None

        horizontal_segments = []
        vertical_segments = []
        connected_pairs = set()

        def add_segment(segment_bucket, player_a, player_b):
            if player_a["player_id"] == player_b["player_id"]:
                return

            pair_key = tuple(sorted((player_a["player_id"], player_b["player_id"])))
            if pair_key in connected_pairs:
                return

            connected_pairs.add(pair_key)
            segment_bucket.append((player_a["screen_position"], player_b["screen_position"]))

        for line_group in line_groups:
            for player_a, player_b in self.collect_horizontal_pairs(line_group):
                add_segment(horizontal_segments, player_a, player_b)

        for current_line, next_line in zip(line_groups, line_groups[1:]):
            for player_a, player_b in self.collect_vertical_pairs(current_line, next_line):
                add_segment(vertical_segments, player_a, player_b)

        if len(horizontal_segments) == 0 and len(vertical_segments) == 0:
            return None

        return {
            "horizontal_segments": horizontal_segments,
            "vertical_segments": vertical_segments,
        }

    def build_team_tactical_overlays(self, tracks, team_id, formation_details):
        overlays = []

        for frame_players in tracks["players"]:
            overlays.append(
                self.build_frame_tactical_overlay(
                    frame_players,
                    team_id,
                    formation_details
                )
            )

        return overlays

    def estimate_formation(self, tracks, team_id):
        avg_positions = self.collect_average_positions(tracks, team_id)

        if len(avg_positions) < 7:
            return "Unknown", [], {
                "team_id": team_id,
                "eligible_players": len(avg_positions),
                "reason": "Not enough tracked players with transformed positions."
            }

        avg_positions = sorted(avg_positions, key=lambda x: x[3], reverse=True)[:11]
        goalkeeper, outfield_players, defending_side, reason = self.identify_goalkeeper_and_order(avg_positions)
        if goalkeeper is None:
            return "Unknown", [], {
                "team_id": team_id,
                "eligible_players": len(avg_positions),
                "reason": reason
            }

        line_counts = self.choose_best_partition(outfield_players, defending_side)
        line_counts = self.normalize_line_counts(line_counts)

        if len(line_counts) < 2:
            return "Unknown", outfield_players, {
                "team_id": team_id,
                "eligible_players": len(avg_positions),
                "goalkeeper_id": goalkeeper[0],
                "defending_side": defending_side,
                "stable_player_ids": [player[0] for player in avg_positions],
                "reason": "Could not separate outfield players into multiple lines."
            }

        formation = "-".join(str(c) for c in line_counts)
        return formation, outfield_players, {
            "team_id": team_id,
            "eligible_players": len(avg_positions),
            "goalkeeper_id": goalkeeper[0],
            "defending_side": defending_side,
            "stable_player_ids": [player[0] for player in avg_positions],
            "line_counts": line_counts
        }
