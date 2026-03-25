import os
import numpy as np
import matplotlib.pyplot as plt


class HeatmapAnalyzer:
    def __init__(self, pitch_length=105, pitch_width=68, grid_size=(50, 35)):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.grid_size = grid_size  # (x_bins, y_bins)

    def _collect_team_positions(self, tracks, team_id):
        positions = []

        for frame_players in tracks["players"]:
            for _, player in frame_players.items():
                if player.get("team") != team_id:
                    continue

                pos = player.get("position_transformed")
                if pos is None:
                    continue

                x, y = pos
                if x is None or y is None:
                    continue

                positions.append((float(x), float(y)))

        return positions

    def _collect_player_positions(self, tracks, player_id):
        positions = []

        for frame_players in tracks["players"]:
            if player_id not in frame_players:
                continue

            pos = frame_players[player_id].get("position_transformed")
            if pos is None:
                continue

            x, y = pos
            if x is None or y is None:
                continue

            positions.append((float(x), float(y)))

        return positions

    def _build_heatmap(self, positions):
        x_bins, y_bins = self.grid_size
        heatmap = np.zeros((y_bins, x_bins), dtype=np.float32)

        if len(positions) == 0:
            return heatmap

        for x, y in positions:
            # clamp to pitch bounds
            x = max(0, min(self.pitch_length, x))
            y = max(0, min(self.pitch_width, y))

            x_idx = min(int((x / self.pitch_length) * x_bins), x_bins - 1)
            y_idx = min(int((y / self.pitch_width) * y_bins), y_bins - 1)

            heatmap[y_idx, x_idx] += 1

        return heatmap

    def _draw_pitch(self, ax):
        ax.set_xlim(0, self.pitch_length)
        ax.set_ylim(0, self.pitch_width)
        ax.set_facecolor("#2e7d32")

        # Outer boundaries
        ax.plot([0, self.pitch_length], [0, 0], color="white")
        ax.plot([0, self.pitch_length], [self.pitch_width, self.pitch_width], color="white")
        ax.plot([0, 0], [0, self.pitch_width], color="white")
        ax.plot([self.pitch_length, self.pitch_length], [0, self.pitch_width], color="white")

        # Halfway line
        ax.plot(
            [self.pitch_length / 2, self.pitch_length / 2],
            [0, self.pitch_width],
            color="white"
        )

        # Center circle
        center_circle = plt.Circle(
            (self.pitch_length / 2, self.pitch_width / 2),
            9.15,
            color="white",
            fill=False
        )
        ax.add_patch(center_circle)

        # Left penalty area
        ax.plot([16.5, 16.5], [13.84, self.pitch_width - 13.84], color="white")
        ax.plot([0, 16.5], [13.84, 13.84], color="white")
        ax.plot([0, 16.5], [self.pitch_width - 13.84, self.pitch_width - 13.84], color="white")

        # Right penalty area
        ax.plot(
            [self.pitch_length - 16.5, self.pitch_length - 16.5],
            [13.84, self.pitch_width - 13.84],
            color="white"
        )
        ax.plot(
            [self.pitch_length, self.pitch_length - 16.5],
            [13.84, 13.84],
            color="white"
        )
        ax.plot(
            [self.pitch_length, self.pitch_length - 16.5],
            [self.pitch_width - 13.84, self.pitch_width - 13.84],
            color="white"
        )

        ax.set_xticks([])
        ax.set_yticks([])

    def save_team_heatmap(self, tracks, team_id, output_path):
        positions = self._collect_team_positions(tracks, team_id)
        heatmap = self._build_heatmap(positions)

        fig, ax = plt.subplots(figsize=(12, 8))
        self._draw_pitch(ax)

        ax.imshow(
            heatmap,
            extent=[0, self.pitch_length, 0, self.pitch_width],
            origin="lower",
            cmap="hot",
            alpha=0.65,
            aspect="auto"
        )

        ax.set_title(f"Team {team_id} Heatmap", color="black", fontsize=16)
        fig.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def save_player_heatmap(self, tracks, player_id, output_path):
        positions = self._collect_player_positions(tracks, player_id)
        heatmap = self._build_heatmap(positions)

        fig, ax = plt.subplots(figsize=(12, 8))
        self._draw_pitch(ax)

        ax.imshow(
            heatmap,
            extent=[0, self.pitch_length, 0, self.pitch_width],
            origin="lower",
            cmap="hot",
            alpha=0.65,
            aspect="auto"
        )

        ax.set_title(f"Player {player_id} Heatmap", color="black", fontsize=16)
        fig.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def get_most_active_players(self, tracks, top_n=2):
        counts = {}

        for frame_players in tracks["players"]:
            for player_id, player in frame_players.items():
                pos = player.get("position_transformed")
                if pos is None:
                    continue

                counts[player_id] = counts.get(player_id, 0) + 1

        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_counts[:top_n]

    def get_ball_source_breakdown(self, tracks):
        source_counts = {
            "detected": 0,
            "interpolated": 0,
            "predicted": 0,
            "missing": 0,
        }

        for ball_dict in tracks["ball"]:
            ball_info = ball_dict.get(1)
            if ball_info is None:
                source_counts["missing"] += 1
                continue

            source_name = ball_info.get("source", "detected")
            source_counts[source_name] = source_counts.get(source_name, 0) + 1

        return source_counts

    def save_match_summary(
        self,
        output_path,
        team_ball_control,
        team_pass_count,
        team_threat_per_frame,
        tracks
    ):
        team_1_frames = sum(1 for x in team_ball_control if x == 1)
        team_2_frames = sum(1 for x in team_ball_control if x == 2)
        total = team_1_frames + team_2_frames

        if total == 0:
            team_1_possession = 0
            team_2_possession = 0
        else:
            team_1_possession = 100 * team_1_frames / total
            team_2_possession = 100 * team_2_frames / total

        high_threat = sum(1 for x in team_threat_per_frame if x["label"] == "High Threat")
        medium_threat = sum(1 for x in team_threat_per_frame if x["label"] == "Medium Threat")
        low_threat = sum(1 for x in team_threat_per_frame if x["label"] == "Low Threat")

        threat_by_team = {
            1: {"High Threat": 0, "Medium Threat": 0, "Low Threat": 0},
            2: {"High Threat": 0, "Medium Threat": 0, "Low Threat": 0},
        }
        for frame_num, threat_info in enumerate(team_threat_per_frame):
            if frame_num >= len(team_ball_control):
                break

            team_id = int(team_ball_control[frame_num])
            if team_id not in threat_by_team:
                continue

            label = threat_info.get("label", "Low Threat")
            if label in threat_by_team[team_id]:
                threat_by_team[team_id][label] += 1

        ball_source_counts = self.get_ball_source_breakdown(tracks)
        total_ball_frames = len(tracks["ball"])
        tracked_ball_frames = total_ball_frames - ball_source_counts["missing"]
        avg_players_per_frame = (
            float(np.mean([len(frame_players) for frame_players in tracks["players"]]))
            if tracks["players"] else 0.0
        )
        unique_player_ids = len({
            player_id
            for frame_players in tracks["players"]
            for player_id in frame_players
        })

        active_players = self.get_most_active_players(tracks, top_n=3)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("SOCCER ANALYTICS MATCH SUMMARY\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"Team 1 Possession: {team_1_possession:.2f}%\n")
            f.write(f"Team 2 Possession: {team_2_possession:.2f}%\n\n")

            f.write(f"Team 1 Estimated Passes: {team_pass_count.get(1, 0)}\n")
            f.write(f"Team 2 Estimated Passes: {team_pass_count.get(2, 0)}\n\n")

            f.write(f"High Threat Frames: {high_threat}\n")
            f.write(f"Medium Threat Frames: {medium_threat}\n")
            f.write(f"Low Threat Frames: {low_threat}\n\n")

            f.write(f"Average Tracked Players Per Frame: {avg_players_per_frame:.2f}\n")
            f.write(f"Unique Stable Player IDs: {unique_player_ids}\n")
            f.write(f"Ball Tracking Coverage: {tracked_ball_frames}/{total_ball_frames} frames\n\n")

            f.write("Threat By Possession Team:\n")
            for team_id in [1, 2]:
                team_threat = threat_by_team[team_id]
                f.write(
                    f"  - Team {team_id}: "
                    f"High {team_threat['High Threat']}, "
                    f"Medium {team_threat['Medium Threat']}, "
                    f"Low {team_threat['Low Threat']}\n"
                )
            f.write("\n")

            f.write("Ball Source Breakdown:\n")
            for source_name in ["detected", "interpolated", "predicted", "missing"]:
                f.write(f"  - {source_name.title()}: {ball_source_counts.get(source_name, 0)}\n")
            f.write("\n")

            f.write("Most Active Players:\n")
            for player_id, count in active_players:
                f.write(f"  - Player {player_id}: {count} tracked frames\n")
