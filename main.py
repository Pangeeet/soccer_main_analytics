import argparse
import os

from utils import read_video, save_video, get_center_of_bbox, get_foot_position, measure_distance
from trackers import Tracker
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from threat_analyzer.threat_analyzer import ThreatAnalyzer
from view_transformer import ViewTransformer
from formation_analyzer import FormationAnalyzer
from heatmap_analyzer import HeatmapAnalyzer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_PATH = os.path.join(BASE_DIR, "field_calibrations.json")
TRACKER_MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
TEAM_IDS = (1, 2)
POSSESSION_INERTIA = 8
RESET_TOUCH_AFTER = 3
PASS_MAX_GAP_SECONDS = 1.25
PASS_MIN_DISTANCE_METERS = 2.5
PASS_MIN_DISTANCE_PIXELS_FALLBACK = 30
PASS_COOLDOWN_SECONDS = 0.40
PASS_EVENT_DURATION_SECONDS = 1.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze a soccer video and generate an annotated output video plus reports."
    )
    parser.add_argument(
        "video",
        nargs="?",
        default="test_8.mp4",
        help="Video filename inside input_videos, or a full path to a video file."
    )
    return parser.parse_args()


def resolve_video_path(video_argument):
    if os.path.isfile(video_argument):
        return os.path.abspath(video_argument)

    candidate_path = os.path.join(BASE_DIR, "input_videos", video_argument)
    if os.path.isfile(candidate_path):
        return candidate_path

    raise FileNotFoundError(
        f"Video not found: '{video_argument}'. Put it in 'input_videos' or pass a full file path."
    )


def build_output_paths(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return {
        "video_name": video_name,
        "track_stub": os.path.join(BASE_DIR, "stubs", f"{video_name}_track_stubs.pkl"),
        "camera_stub": os.path.join(BASE_DIR, "stubs", f"{video_name}_camera_movement_stub.pkl"),
        "report": os.path.join(BASE_DIR, "output_reports", f"{video_name}_team_formations.txt"),
        "summary": os.path.join(BASE_DIR, "output_reports", f"{video_name}_match_summary.txt"),
        "diagnostics": os.path.join(BASE_DIR, "output_reports", f"{video_name}_analysis_diagnostics.txt"),
        "heatmap_dir": os.path.join(BASE_DIR, "output_reports", f"{video_name}_heatmaps"),
        "video": os.path.join(BASE_DIR, "output_videos", f"{video_name}_output.avi"),
    }


def ensure_parent_directories(paths):
    for path in paths:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)


def filter_tracks_to_pitch(tracks):
    for object_name in ["players", "referees"]:
        for frame_num, frame_tracks in enumerate(tracks[object_name]):
            filtered_tracks = {
                track_id: track_info
                for track_id, track_info in frame_tracks.items()
                if track_info.get("position_transformed") is not None
            }
            tracks[object_name][frame_num] = filtered_tracks


def build_pass_summary(team_pass_count, team_pass_links):
    summary = {
        "team_pass_count": dict(team_pass_count),
        "top_links": {},
    }

    for team_id, link_counts in team_pass_links.items():
        ranked_links = sorted(
            link_counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1])
        )
        summary["top_links"][team_id] = [
            (sender_id, receiver_id, count)
            for (sender_id, receiver_id), count in ranked_links[:5]
        ]

    return summary


def seconds_to_frames(video_fps, seconds, minimum=1):
    fps = max(float(video_fps or 0.0), 1.0)
    return max(minimum, int(round(fps * float(seconds))))


def summarize_ball_tracking(ball_tracks):
    source_counts = {
        "detected": 0,
        "interpolated": 0,
        "predicted": 0,
        "missing": 0,
    }
    confidence_values = []
    transformed_count = 0

    for ball_dict in ball_tracks:
        ball_info = ball_dict.get(1)
        if ball_info is None:
            source_counts["missing"] += 1
            continue

        source_name = ball_info.get("source", "detected")
        source_counts[source_name] = source_counts.get(source_name, 0) + 1

        if ball_info.get("position_transformed") is not None:
            transformed_count += 1

        confidence = ball_info.get("confidence")
        if confidence is not None and confidence > 0:
            confidence_values.append(float(confidence))

    total_frames = len(ball_tracks)
    tracked_frames = total_frames - source_counts["missing"]
    return {
        "source_counts": source_counts,
        "tracked_frames": tracked_frames,
        "total_frames": total_frames,
        "tracked_pct": (100.0 * tracked_frames / total_frames) if total_frames > 0 else 0.0,
        "transformed_pct": (100.0 * transformed_count / total_frames) if total_frames > 0 else 0.0,
        "avg_confidence": float(np.mean(confidence_values)) if confidence_values else 0.0,
    }


def build_tracking_diagnostics(tracking_data, possession_data, formation_results):
    video_frames = tracking_data["video_frames"]
    video_metadata = tracking_data["video_metadata"]
    tracks = tracking_data["tracks"]
    view_transformer = tracking_data["view_transformer"]

    player_counts_per_frame = [len(frame_players) for frame_players in tracks["players"]]
    avg_players_per_frame = float(np.mean(player_counts_per_frame)) if player_counts_per_frame else 0.0
    median_players_per_frame = float(np.median(player_counts_per_frame)) if player_counts_per_frame else 0.0
    unique_player_ids = len({
        player_id
        for frame_players in tracks["players"]
        for player_id in frame_players
    })
    team_presence = {
        team_id: float(np.mean([
            sum(1 for player in frame_players.values() if player.get("team") == team_id)
            for frame_players in tracks["players"]
        ])) if tracks["players"] else 0.0
        for team_id in TEAM_IDS
    }

    possession_known_frames = sum(1 for team_id in possession_data["team_ball_control"] if team_id in TEAM_IDS)
    ball_summary = summarize_ball_tracking(tracks["ball"])

    formation_summary = {}
    for team_id in TEAM_IDS:
        team_result = formation_results.get(team_id, {})
        details = team_result.get("details", {})
        shape_summary = team_result.get("shape_summary", {})
        formation_summary[team_id] = {
            "formation": team_result.get("formation", "Unknown"),
            "eligible_players": details.get("eligible_players", 0),
            "defending_side": details.get("defending_side", "Unknown"),
            "shape_label": shape_summary.get("shape_label", "Unknown"),
            "valid_shape_frames": shape_summary.get("valid_frames", 0),
        }

    return {
        "video_name": tracking_data["video_name"],
        "frame_count": len(video_frames),
        "source_fps": float(video_metadata.get("fps", 24.0)),
        "frame_width": int(video_metadata.get("frame_width", 0)),
        "frame_height": int(video_metadata.get("frame_height", 0)),
        "pitch_length": float(view_transformer.pitch_length),
        "pitch_width": float(view_transformer.pitch_width),
        "calibration_name": view_transformer.calibration_name,
        "avg_players_per_frame": avg_players_per_frame,
        "median_players_per_frame": median_players_per_frame,
        "unique_player_ids": unique_player_ids,
        "team_presence": team_presence,
        "possession_known_frames": possession_known_frames,
        "possession_known_pct": (
            100.0 * possession_known_frames / len(possession_data["team_ball_control"])
            if len(possession_data["team_ball_control"]) > 0 else 0.0
        ),
        "ball_summary": ball_summary,
        "formation_summary": formation_summary,
    }


def save_diagnostics_report(path, diagnostics):
    ball_summary = diagnostics["ball_summary"]
    source_counts = ball_summary["source_counts"]

    with open(path, "w", encoding="utf-8") as report_file:
        report_file.write("SOCCER ANALYTICS DIAGNOSTICS\n")
        report_file.write("============================\n\n")
        report_file.write(f"Video: {diagnostics['video_name']}\n")
        report_file.write(
            f"Resolution / FPS: {diagnostics['frame_width']}x{diagnostics['frame_height']} @ "
            f"{diagnostics['source_fps']:.2f}\n"
        )
        report_file.write(f"Frames processed: {diagnostics['frame_count']}\n")
        report_file.write(f"Calibration profile: {diagnostics['calibration_name']}\n")
        report_file.write(
            f"Pitch dimensions: {diagnostics['pitch_length']:.1f} x {diagnostics['pitch_width']:.1f}\n\n"
        )

        report_file.write(f"Average tracked players per frame: {diagnostics['avg_players_per_frame']:.2f}\n")
        report_file.write(f"Median tracked players per frame: {diagnostics['median_players_per_frame']:.2f}\n")
        report_file.write(f"Unique stable player IDs: {diagnostics['unique_player_ids']}\n")
        for team_id in TEAM_IDS:
            report_file.write(
                f"Average Team {team_id} players per frame: "
                f"{diagnostics['team_presence'][team_id]:.2f}\n"
            )
        report_file.write("\n")

        report_file.write(
            f"Possession assigned on {diagnostics['possession_known_frames']} frames "
            f"({diagnostics['possession_known_pct']:.1f}%)\n"
        )
        report_file.write(
            f"Ball tracked on {ball_summary['tracked_frames']} / {ball_summary['total_frames']} frames "
            f"({ball_summary['tracked_pct']:.1f}%)\n"
        )
        report_file.write(
            f"Ball projected to pitch on {ball_summary['transformed_pct']:.1f}% of frames\n"
        )
        report_file.write(f"Average ball confidence: {ball_summary['avg_confidence']:.3f}\n")
        report_file.write("Ball source counts:\n")
        for source_name in ["detected", "interpolated", "predicted", "missing"]:
            report_file.write(f"  - {source_name.title()}: {source_counts.get(source_name, 0)}\n")
        report_file.write("\n")

        report_file.write("Formation snapshot:\n")
        for team_id in TEAM_IDS:
            team_summary = diagnostics["formation_summary"][team_id]
            report_file.write(
                f"  - Team {team_id}: {team_summary['formation']} | "
                f"{team_summary['shape_label']} | "
                f"eligible players {team_summary['eligible_players']} | "
                f"defending side {team_summary['defending_side']} | "
                f"valid shape frames {team_summary['valid_shape_frames']}\n"
            )


def save_formation_report(path, team_results, pass_summary=None):
    with open(path, "w", encoding="utf-8") as report_file:
        report_file.write("TEAM FORMATION REPORT\n")
        report_file.write("=====================\n\n")

        for team_id, formation, avg_positions, details, shape_summary in team_results:
            report_file.write(f"Team {team_id} estimated formation: {formation}\n")
            report_file.write(f"Eligible tracked players: {details.get('eligible_players', 0)}\n")

            goalkeeper_id = details.get("goalkeeper_id")
            if goalkeeper_id is not None:
                report_file.write(f"Likely goalkeeper player ID: {goalkeeper_id}\n")

            line_counts = details.get("line_counts")
            if line_counts:
                report_file.write(f"Detected outfield lines: {line_counts}\n")

            defending_side = details.get("defending_side")
            if defending_side:
                report_file.write(f"Defending side: {defending_side}\n")

            reason = details.get("reason")
            if reason:
                report_file.write(f"Note: {reason}\n")

            shape_label = shape_summary.get("shape_label")
            if shape_label:
                report_file.write(f"Shape profile: {shape_label}\n")

            if "avg_width_pct" in shape_summary:
                report_file.write(f"Average width: {shape_summary['avg_width_pct']:.1f}% of pitch width\n")
                report_file.write(f"Average length: {shape_summary['avg_length_pct']:.1f}% of pitch length\n")
                report_file.write(f"Average compactness: {shape_summary['avg_compactness_pct']:.1f}% of pitch area\n")
                report_file.write(f"Average defensive line height: {shape_summary['avg_line_height_pct']:.1f}% upfield\n")
                report_file.write(f"Average team centroid progress: {shape_summary['avg_centroid_progress_pct']:.1f}% upfield\n")
                report_file.write(f"Average players used per frame: {shape_summary['avg_player_count']:.1f}\n")
                report_file.write(f"Valid team-shape frames: {shape_summary['valid_frames']}\n")

            shape_reason = shape_summary.get("reason")
            if shape_reason:
                report_file.write(f"Shape note: {shape_reason}\n")

            if avg_positions:
                report_file.write("Outfield average positions (player_id, x, y):\n")
                for player_id, avg_x, avg_y, _ in avg_positions:
                    report_file.write(f"  - {player_id}: ({avg_x:.2f}, {avg_y:.2f})\n")

            report_file.write("\n")

        if pass_summary is not None:
            report_file.write("PASS SUMMARY\n")
            report_file.write("============\n\n")

            team_pass_count = pass_summary.get("team_pass_count", {})
            top_links = pass_summary.get("top_links", {})

            for team_id in [1, 2]:
                report_file.write(f"Team {team_id} completed passes: {team_pass_count.get(team_id, 0)}\n")

                ranked_links = top_links.get(team_id, [])
                if ranked_links:
                    report_file.write("Top detected pass links (sender -> receiver):\n")
                    for sender_id, receiver_id, count in ranked_links:
                        report_file.write(f"  - {sender_id} -> {receiver_id}: {count}\n")
                else:
                    report_file.write("Top detected pass links: none\n")

                report_file.write("\n")


def prepare_run(video_argument):
    video_path = resolve_video_path(video_argument)
    output_paths = build_output_paths(video_path)
    ensure_parent_directories([
        output_paths["track_stub"],
        output_paths["camera_stub"],
        output_paths["report"],
        output_paths["summary"],
        output_paths["diagnostics"],
        output_paths["video"],
    ])
    os.makedirs(output_paths["heatmap_dir"], exist_ok=True)

    return video_path, output_paths


def run_tracking_pipeline(video_path, output_paths):
    video_frames, video_metadata = read_video(video_path, return_metadata=True)
    if not video_frames:
        raise ValueError(f"No frames could be read from video: {video_path}")

    tracker = Tracker(TRACKER_MODEL_PATH)

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=False,
        stub_path=output_paths["track_stub"]
    )

    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    tracker.add_position_to_tracks(tracks)

    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=False,
        stub_path=output_paths["camera_stub"]
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(
        tracks,
        camera_movement_per_frame
    )

    view_transformer = ViewTransformer(
        calibration_path=CALIBRATION_PATH,
        video_key=output_paths["video_name"]
    )
    view_transformer.add_transformed_position_to_tracks(tracks)
    filter_tracks_to_pitch(tracks)
    tracker.stabilize_player_track_ids(tracks)

    return {
        "video_name": output_paths["video_name"],
        "video_frames": video_frames,
        "video_metadata": video_metadata,
        "tracker": tracker,
        "tracks": tracks,
        "camera_movement_estimator": camera_movement_estimator,
        "camera_movement_per_frame": camera_movement_per_frame,
        "view_transformer": view_transformer,
    }


def assign_player_teams(video_frames, tracks):
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames, tracks["players"])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track["bbox"],
                player_id
            )
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    return team_assigner


def analyze_team_formations(tracks, view_transformer):
    formation_analyzer = FormationAnalyzer(
        pitch_length=view_transformer.pitch_length,
        pitch_width=view_transformer.pitch_width
    )

    formation_results = {}
    for team_id in TEAM_IDS:
        formation, avg_positions, details = formation_analyzer.estimate_formation(tracks, team_id=team_id)
        shape_summary, shape_frames = formation_analyzer.analyze_team_shape(
            tracks,
            team_id=team_id,
            formation_details=details
        )
        tactical_lines = formation_analyzer.build_team_tactical_overlays(
            tracks,
            team_id=team_id,
            formation_details=details
        )
        formation_results[team_id] = {
            "formation": formation,
            "avg_positions": avg_positions,
            "details": details,
            "shape_summary": shape_summary,
            "shape_frames": shape_frames,
            "tactical_lines": tactical_lines,
        }

    return formation_results


def initialize_possession_state():
    return {
        "current_possession_team": 0,
        "candidate_team": 0,
        "candidate_count": 0,
        "team_ball_control": [],
        "team_touch_count": {team_id: 0 for team_id in TEAM_IDS},
        "team_touch_count_per_frame": [],
        "team_pass_count": {team_id: 0 for team_id in TEAM_IDS},
        "team_pass_count_per_frame": [],
        "team_pass_links": {team_id: {} for team_id in TEAM_IDS},
        "pass_events_per_frame": [],
        "last_touch_player": None,
        "last_touch_team": 0,
        "last_touch_position": None,
        "last_touch_position_space": None,
        "no_control_frames": 0,
        "last_pass_frame": -999,
        "active_pass_event": None,
        "active_pass_event_until": -1,
    }


def expire_pass_event(frame_num, possession_state):
    if frame_num > possession_state["active_pass_event_until"]:
        possession_state["active_pass_event"] = None


def clear_ball_carrier_flags(player_track):
    for track in player_track.values():
        track["has_ball"] = False


def get_player_reference_position(player_track_info):
    transformed_position = player_track_info.get("position_transformed")
    if transformed_position is not None and len(transformed_position) >= 2:
        return (float(transformed_position[0]), float(transformed_position[1])), "transformed"

    foot_position = get_foot_position(player_track_info["bbox"])
    return foot_position, "image"


def assign_ball_carrier(player_assigner, player_track, ball_track):
    ball_info = ball_track.get(1, {})
    ball_bbox = ball_info.get("bbox")
    ball_source = ball_info.get("source", "detected")

    if ball_source == "predicted":
        previous_player = player_assigner.last_assigned_player
        if previous_player in player_track and ball_bbox is not None:
            ball_position = get_center_of_bbox(ball_bbox)
            previous_player_bbox = player_track[previous_player]["bbox"]
            if player_assigner.is_ball_close_enough(previous_player_bbox, ball_position):
                return previous_player
        return -1

    return player_assigner.assign_ball_to_player(player_track, ball_bbox)


def should_count_pass(
    frame_num,
    assigned_player,
    detected_team,
    touch_gap_frames,
    possession_state,
    pass_max_gap_frames,
    pass_cooldown_frames
):
    return (
        possession_state["last_touch_player"] is not None and
        assigned_player != possession_state["last_touch_player"] and
        detected_team in TEAM_IDS and
        detected_team == possession_state["last_touch_team"] and
        possession_state["last_touch_position"] is not None and
        touch_gap_frames <= pass_max_gap_frames and
        (frame_num - possession_state["last_pass_frame"]) >= pass_cooldown_frames
    )


def register_completed_pass(
    frame_num,
    assigned_player,
    detected_team,
    current_position,
    current_position_space,
    possession_state,
    pass_event_duration_frames
):
    previous_position = possession_state["last_touch_position"]
    previous_position_space = possession_state["last_touch_position_space"]
    pass_distance = measure_distance(previous_position, current_position)

    if previous_position_space == "transformed" and current_position_space == "transformed":
        if pass_distance < PASS_MIN_DISTANCE_METERS:
            return
    elif pass_distance < PASS_MIN_DISTANCE_PIXELS_FALLBACK:
        return

    possession_state["team_pass_count"][detected_team] += 1
    pass_link = (int(possession_state["last_touch_player"]), int(assigned_player))
    possession_state["team_pass_links"][detected_team][pass_link] = (
        possession_state["team_pass_links"][detected_team].get(pass_link, 0) + 1
    )
    possession_state["active_pass_event"] = {
        "team_id": int(detected_team),
        "from_player": int(possession_state["last_touch_player"]),
        "to_player": int(assigned_player),
    }
    possession_state["active_pass_event_until"] = frame_num + pass_event_duration_frames
    possession_state["last_pass_frame"] = frame_num


def update_touch_and_pass_state(
    frame_num,
    player_track,
    assigned_player,
    possession_state,
    pass_max_gap_frames,
    pass_cooldown_frames,
    pass_event_duration_frames
):
    if assigned_player == -1:
        possession_state["no_control_frames"] += 1
        return 0

    player_track[assigned_player]["has_ball"] = True
    detected_team = player_track[assigned_player]["team"]
    current_position, current_position_space = get_player_reference_position(player_track[assigned_player])
    touch_gap_frames = possession_state["no_control_frames"]
    is_new_touch = (
        assigned_player != possession_state["last_touch_player"] or
        touch_gap_frames >= RESET_TOUCH_AFTER
    )

    if is_new_touch and detected_team in TEAM_IDS:
        possession_state["team_touch_count"][detected_team] += 1

    if is_new_touch and should_count_pass(
        frame_num,
        assigned_player,
        detected_team,
        touch_gap_frames,
        possession_state,
        pass_max_gap_frames,
        pass_cooldown_frames
    ):
        register_completed_pass(
            frame_num,
            assigned_player,
            detected_team,
            current_position,
            current_position_space,
            possession_state,
            pass_event_duration_frames
        )

    possession_state["last_touch_player"] = assigned_player
    possession_state["last_touch_team"] = detected_team
    possession_state["last_touch_position"] = current_position
    possession_state["last_touch_position_space"] = current_position_space
    possession_state["no_control_frames"] = 0

    return detected_team


def update_possession_control(detected_team, possession_state):
    if detected_team == 0:
        possession_state["team_ball_control"].append(possession_state["current_possession_team"])
        return

    if possession_state["current_possession_team"] == 0:
        possession_state["current_possession_team"] = detected_team
        possession_state["candidate_team"] = 0
        possession_state["candidate_count"] = 0
        possession_state["team_ball_control"].append(possession_state["current_possession_team"])
        return

    if detected_team == possession_state["current_possession_team"]:
        possession_state["candidate_team"] = 0
        possession_state["candidate_count"] = 0
        possession_state["team_ball_control"].append(possession_state["current_possession_team"])
        return

    if detected_team == possession_state["candidate_team"]:
        possession_state["candidate_count"] += 1
    else:
        possession_state["candidate_team"] = detected_team
        possession_state["candidate_count"] = 1

    if possession_state["candidate_count"] >= POSSESSION_INERTIA:
        possession_state["current_possession_team"] = possession_state["candidate_team"]
        possession_state["candidate_team"] = 0
        possession_state["candidate_count"] = 0

    possession_state["team_ball_control"].append(possession_state["current_possession_team"])


def record_frame_state(possession_state):
    possession_state["team_touch_count_per_frame"].append(possession_state["team_touch_count"].copy())
    possession_state["team_pass_count_per_frame"].append(possession_state["team_pass_count"].copy())
    possession_state["pass_events_per_frame"].append(possession_state["active_pass_event"])


def analyze_possession_and_passes(tracks, video_fps=30.0):
    player_assigner = PlayerBallAssigner()
    possession_state = initialize_possession_state()
    pass_max_gap_frames = seconds_to_frames(video_fps, PASS_MAX_GAP_SECONDS, minimum=8)
    pass_cooldown_frames = seconds_to_frames(video_fps, PASS_COOLDOWN_SECONDS, minimum=4)
    pass_event_duration_frames = seconds_to_frames(video_fps, PASS_EVENT_DURATION_SECONDS, minimum=12)

    for frame_num, player_track in enumerate(tracks["players"]):
        expire_pass_event(frame_num, possession_state)
        clear_ball_carrier_flags(player_track)

        assigned_player = assign_ball_carrier(
            player_assigner,
            player_track,
            tracks["ball"][frame_num]
        )
        detected_team = update_touch_and_pass_state(
            frame_num,
            player_track,
            assigned_player,
            possession_state,
            pass_max_gap_frames,
            pass_cooldown_frames,
            pass_event_duration_frames
        )
        update_possession_control(detected_team, possession_state)
        record_frame_state(possession_state)

    possession_state["team_ball_control"] = np.array(possession_state["team_ball_control"])
    possession_state["pass_summary"] = build_pass_summary(
        possession_state["team_pass_count"],
        possession_state["team_pass_links"]
    )

    return possession_state


def analyze_threat_levels(tracks, team_ball_control, view_transformer):
    threat_analyzer = ThreatAnalyzer(
        pitch_length=view_transformer.pitch_length,
        pitch_width=view_transformer.pitch_width
    )
    team_threat_per_frame = []

    for frame_num in range(len(tracks["ball"])):
        possession_team = int(team_ball_control[frame_num]) if frame_num < len(team_ball_control) else 0

        ball_dict = tracks["ball"][frame_num]
        ball_position = None

        if 1 in ball_dict:
            ball_position = ball_dict[1].get("position_transformed")

        threat_info = threat_analyzer.get_threat_info(ball_position, possession_team)
        team_threat_per_frame.append(threat_info)

    return team_threat_per_frame


def print_touch_pass_threat_summary(possession_data, team_threat_per_frame):
    print("\n===== TOUCH SUMMARY =====")
    for team_id in TEAM_IDS:
        print(f"Team {team_id} touches: {possession_data['team_touch_count'][team_id]}")

    print("\n===== PASS SUMMARY =====")
    for team_id in TEAM_IDS:
        print(f"Team {team_id} completed passes: {possession_data['team_pass_count'][team_id]}")

    print("\n===== THREAT SUMMARY =====")
    high_threat_frames = sum(1 for x in team_threat_per_frame if x["label"] == "High Threat")
    medium_threat_frames = sum(1 for x in team_threat_per_frame if x["label"] == "Medium Threat")
    low_threat_frames = sum(1 for x in team_threat_per_frame if x["label"] == "Low Threat")

    print(f"High Threat frames: {high_threat_frames}")
    print(f"Medium Threat frames: {medium_threat_frames}")
    print(f"Low Threat frames: {low_threat_frames}")


def print_tracking_diagnostics(tracking_data, possession_data):
    video_frames = tracking_data["video_frames"]
    tracks = tracking_data["tracks"]
    view_transformer = tracking_data["view_transformer"]

    print("len(video_frames):", len(video_frames))
    print("len(tracks['players']):", len(tracks["players"]))
    print("len(tracks['ball']):", len(tracks["ball"]))
    print("len(tracks['referees']):", len(tracks["referees"]))
    print("len(team_ball_control):", len(possession_data["team_ball_control"]))
    print("len(camera_movement_per_frame):", len(tracking_data["camera_movement_per_frame"]))
    print("source fps:", tracking_data["video_metadata"]["fps"])
    print("field calibration profile:", view_transformer.calibration_name)
    print("pitch width/length:", view_transformer.pitch_width, view_transformer.pitch_length)


def print_formation_summary(formation_results):
    print("\n===== FORMATION SUMMARY =====")
    for team_id in TEAM_IDS:
        team_result = formation_results[team_id]
        details = team_result["details"]
        shape_summary = team_result["shape_summary"]

        print(f"Team {team_id} estimated formation: {team_result['formation']}")
        if details.get("reason"):
            print(f"Team {team_id} note: {details['reason']}")
        print(f"Team {team_id} shape profile: {shape_summary.get('shape_label', 'Unknown')}")
        if "avg_width_pct" in shape_summary:
            print(
                f"Team {team_id} width/length/compactness/line: "
                f"{shape_summary['avg_width_pct']:.1f}% / "
                f"{shape_summary['avg_length_pct']:.1f}% / "
                f"{shape_summary['avg_compactness_pct']:.1f}% / "
                f"{shape_summary['avg_line_height_pct']:.1f}%"
            )


def save_analysis_report(output_paths, formation_results, pass_summary):
    team_results = [
        (
            team_id,
            formation_results[team_id]["formation"],
            formation_results[team_id]["avg_positions"],
            formation_results[team_id]["details"],
            formation_results[team_id]["shape_summary"],
        )
        for team_id in TEAM_IDS
    ]
    save_formation_report(output_paths["report"], team_results, pass_summary=pass_summary)


def save_extra_analysis_artifacts(output_paths, tracking_data, possession_data, team_threat_per_frame):
    heatmap_analyzer = HeatmapAnalyzer(
        pitch_length=tracking_data["view_transformer"].pitch_length,
        pitch_width=tracking_data["view_transformer"].pitch_width
    )
    tracks = tracking_data["tracks"]

    heatmap_analyzer.save_match_summary(
        output_paths["summary"],
        possession_data["team_ball_control"],
        possession_data["team_pass_count"],
        team_threat_per_frame,
        tracks
    )

    for team_id in TEAM_IDS:
        heatmap_analyzer.save_team_heatmap(
            tracks,
            team_id,
            os.path.join(output_paths["heatmap_dir"], f"team_{team_id}_heatmap.png")
        )

    for player_id, _ in heatmap_analyzer.get_most_active_players(tracks, top_n=2):
        heatmap_analyzer.save_player_heatmap(
            tracks,
            player_id,
            os.path.join(output_paths["heatmap_dir"], f"player_{player_id}_heatmap.png")
        )


def save_diagnostics_artifacts(output_paths, tracking_data, possession_data, formation_results):
    diagnostics = build_tracking_diagnostics(
        tracking_data,
        possession_data,
        formation_results
    )
    save_diagnostics_report(output_paths["diagnostics"], diagnostics)


def render_output_video(
    tracking_data,
    possession_data,
    team_threat_per_frame,
    formation_results,
    team_assigner,
    output_paths
):
    tracker = tracking_data["tracker"]
    video_frames = tracking_data["video_frames"]
    tracks = tracking_data["tracks"]
    view_transformer = tracking_data["view_transformer"]

    output_video_frames = tracker.draw_annotations(
        video_frames,
        tracks,
        possession_data["team_ball_control"],
        possession_data["team_touch_count_per_frame"],
        team_pass_count_per_frame=possession_data["team_pass_count_per_frame"],
        team_threat_per_frame=team_threat_per_frame,
        team_formations={
            team_id: formation_results[team_id]["formation"]
            for team_id in TEAM_IDS
        },
        team_shape_per_frame={
            team_id: formation_results[team_id]["shape_frames"]
            for team_id in TEAM_IDS
        },
        team_shape_summaries={
            team_id: formation_results[team_id]["shape_summary"]
            for team_id in TEAM_IDS
        },
        team_tactical_lines={
            team_id: formation_results[team_id]["tactical_lines"]
            for team_id in TEAM_IDS
        },
        team_line_colors={
            1: team_assigner.team_colors.get(1, (255, 255, 255)),
            2: team_assigner.team_colors.get(2, (0, 0, 255)),
        },
        pass_events_per_frame=possession_data["pass_events_per_frame"],
        pitch_dimensions=(view_transformer.pitch_length, view_transformer.pitch_width),
        video_fps=tracking_data["video_metadata"]["fps"],
        copy_frames=False
    )

    output_video_frames = tracking_data["camera_movement_estimator"].draw_camera_movement(
        output_video_frames,
        tracking_data["camera_movement_per_frame"],
        copy_frames=False
    )

    save_video(
        output_video_frames,
        output_paths["video"],
        fps=tracking_data["video_metadata"]["fps"]
    )


def print_output_paths(video_path, output_paths):
    print("\n===== OUTPUT FILES =====")
    print(f"Input video: {video_path}")
    print(f"Annotated video: {output_paths['video']}")
    print(f"Formation report: {output_paths['report']}")
    print(f"Match summary: {output_paths['summary']}")
    print(f"Diagnostics report: {output_paths['diagnostics']}")
    print(f"Heatmaps: {output_paths['heatmap_dir']}")
    print(f"Calibration file: {CALIBRATION_PATH}")


def main():
    args = parse_args()
    video_path, output_paths = prepare_run(args.video)
    tracking_data = run_tracking_pipeline(video_path, output_paths)
    team_assigner = assign_player_teams(
        tracking_data["video_frames"],
        tracking_data["tracks"]
    )
    formation_results = analyze_team_formations(
        tracking_data["tracks"],
        tracking_data["view_transformer"]
    )
    possession_data = analyze_possession_and_passes(
        tracking_data["tracks"],
        video_fps=tracking_data["video_metadata"]["fps"]
    )
    team_threat_per_frame = analyze_threat_levels(
        tracking_data["tracks"],
        possession_data["team_ball_control"],
        tracking_data["view_transformer"]
    )

    print_touch_pass_threat_summary(possession_data, team_threat_per_frame)
    print_tracking_diagnostics(tracking_data, possession_data)
    print_formation_summary(formation_results)
    save_analysis_report(
        output_paths,
        formation_results,
        possession_data["pass_summary"]
    )
    save_extra_analysis_artifacts(
        output_paths,
        tracking_data,
        possession_data,
        team_threat_per_frame
    )
    save_diagnostics_artifacts(
        output_paths,
        tracking_data,
        possession_data,
        formation_results
    )
    render_output_video(
        tracking_data,
        possession_data,
        team_threat_per_frame,
        formation_results,
        team_assigner,
        output_paths
    )
    print_output_paths(video_path, output_paths)


if __name__ == '__main__':
    main()
