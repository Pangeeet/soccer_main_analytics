# Soccer Match Video Analytics

This project analyzes soccer match footage with a YOLOv8-based pipeline and produces both a processed video and supporting reports. It is built around my own workflow for taking a match clip, tracking the players and ball, estimating team behavior, and exporting results that are easier to review.

The repository includes a sample model file and a sample input video, so the default command can run after cloning once the Python dependencies are installed.

## What the project does

- Detects and tracks players, referees, and the ball across the video
- Assigns players to teams based on appearance
- Estimates camera movement and compensates for it
- Projects positions onto a calibrated pitch view
- Tracks ball possession and summarizes pass links
- Estimates team formations and team shape
- Generates match summaries, diagnostics, and heatmaps
- Saves an annotated output video

## Repository structure

- `main.py`: main entry point for the full analysis pipeline
- `input_videos/`: place source videos here
- `output_videos/`: processed videos are saved here
- `output_reports/`: text reports and heatmaps are written here
- `models/`: YOLO model weights used by the tracker
- `stubs/`: cached tracking and camera-movement data
- `field_calibrations.json`: per-video pitch calibration settings
- `training/`: notebooks and dataset files related to training experiments

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Pangeeet/soccer_main_analytics.git
   cd soccer_main_analytics
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure the tracker weights are available in `models/best.pt`.
   The repository already includes this file for the default demo run.

4. The repository also includes `input_videos/test_8.mp4` for the default demo run.
   You can replace it later with your own video or pass a full file path on the command line.

## Run the analysis

The default run uses `test_8.mp4`:

```bash
python main.py
```

You can also analyze a specific file:

```bash
python main.py test_8.mp4
python main.py "C:\path\to\your\video.mp4"
```

## Outputs

After a run, the project can generate:

- an annotated video in `output_videos/`
- a team formation report in `output_reports/`
- a match summary in `output_reports/`
- a diagnostics report in `output_reports/`
- team and player heatmaps in `output_reports/<video_name>_heatmaps/`

## Notes

- Large assets such as raw videos, processed videos, model weights, cached stubs, and generated reports are intentionally excluded from GitHub.
- The `input_videos/`, `output_videos/`, and `output_reports/` folders stay visible in the repository through placeholder note files.
- Field calibration values are stored in `field_calibrations.json`. Add a new entry there when you analyze a video recorded from a different camera angle.

## License

This project is released under the MIT License. See `LICENSE` for details.
