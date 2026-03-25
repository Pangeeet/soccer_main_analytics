class ThreatAnalyzer:
    def __init__(self, pitch_length=105.0, pitch_width=68.0):
        self.pitch_length = max(1.0, float(pitch_length))
        self.pitch_width = max(1.0, float(pitch_width))

    def get_progress_ratio(self, x_position, possession_team):
        progress_ratio = float(x_position) / self.pitch_length
        progress_ratio = min(max(progress_ratio, 0.0), 1.0)

        if possession_team == 2:
            progress_ratio = 1.0 - progress_ratio

        return progress_ratio

    def get_centrality_ratio(self, y_position):
        center_y = self.pitch_width / 2.0
        normalized_offset = abs(float(y_position) - center_y) / max(center_y, 1e-6)
        return 1.0 - min(max(normalized_offset, 0.0), 1.0)

    def get_threat_info(self, ball_position, possession_team):
        """
        ball_position: transformed field position like (x, y)
        possession_team: 1, 2, or 0
        returns: dict with score and label
        """

        if possession_team == 0 or ball_position is None:
            return {"score": 0, "label": "No Threat"}

        x_position, y_position = ball_position
        progress_ratio = self.get_progress_ratio(x_position, possession_team)
        centrality_ratio = self.get_centrality_ratio(y_position)

        score = 0

        if progress_ratio >= 0.45:
            score += 1
        if progress_ratio >= 0.62:
            score += 1
        if progress_ratio >= 0.78:
            score += 2

        if centrality_ratio >= 0.35:
            score += 1
        if progress_ratio >= 0.72 and centrality_ratio >= 0.55:
            score += 1

        if progress_ratio < 0.22:
            label = "Low Threat"
        elif score <= 2:
            label = "Low Threat"
        elif score <= 4:
            label = "Medium Threat"
        else:
            label = "High Threat"

        return {
            "score": score,
            "label": label,
            "progress_ratio": progress_ratio,
            "centrality_ratio": centrality_ratio,
        }
