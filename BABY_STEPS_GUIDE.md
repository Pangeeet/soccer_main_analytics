# Soccer Analysis Project – Baby Steps Guide

This guide is written for you to follow slowly and safely.

---

## 1) What you already had

Your professor's project already does these things:

- detects players
- detects referees
- detects the ball
- tracks objects across frames
- assigns teams by jersey color
- estimates camera movement
- estimates speed and distance
- estimates ball possession

So we do **not** need to build everything from zero.

---

## 2) What I added

I added 3 practical upgrades:

### A. Player position
Each player now gets:
- **pitch coordinates** `(x, y)` from the field transform
- a **basic role label** such as:
  - `GK`
  - `DEF`
  - `MID`
  - `FWD`
  - `AM` (for 4-2-3-1 cases)

### B. Ball position
The ball already existed in the model, but I fixed the order of operations so the project now uses:
- interpolated ball bounding box
- image position
- transformed pitch position

### C. Formation detection
I added a new module that tries to detect common formations:
- `4-4-2`
- `4-3-3`
- `4-2-3-1`
- `3-5-2`
- `3-4-3`
- `5-3-2`
- `4-5-1`

It also returns a **confidence percentage**.

---

## 3) Very important honesty for your report

You asked for:

> determine what is the percentage that it was gonna be accurate or winrate

Here is the academically correct answer:

### What we can measure now
- formation prediction confidence
- how often the detected formation stays stable
- possession percentage
- player positions
- ball positions

### What we cannot claim honestly from this code alone
We **cannot** predict true match **win rate** from formation alone unless we train on a labeled historical dataset such as:
- many matches
- each team's formation
- match outcome (win/draw/loss)
- maybe possession, shots, xG, passes, etc.

So for your submission, say this:

> “This project detects likely team formation and reports a confidence score. A real win-rate model would require supervised training on historical labeled match data.”

That sentence is safe and correct.

---

## 4) Files I changed

### New file
- `formation_analyzer/formation_analyzer.py`

### Changed files
- `main.py`
- `trackers/tracker.py`

---

## 5) How the formation part works in simple words

For each frame:

1. take all players from Team 1
2. take all players from Team 2
3. use transformed field coordinates
4. guess which player is the goalkeeper
5. remove goalkeeper
6. look at the 10 outfield players
7. group them into 3 or 4 tactical lines
8. compare the line sizes to known formation templates

Example:

- line sizes `[4, 4, 2]` → likely `4-4-2`
- line sizes `[4, 3, 3]` → likely `4-3-3`
- line sizes `[4, 2, 3, 1]` → likely `4-2-3-1`

This is a **heuristic detector**, not a deep trained model.

---

## 6) How to run

Open terminal in the project folder and run:

```bash
python main.py
```

The result video is saved in:

```bash
output_videos/output_video11.avi
```

The terminal will also print something like:

```bash
===== FORMATION SUMMARY =====
Team 1: 4-3-3 | average confidence = 78.42%
Team 2: 4-4-2 | average confidence = 73.15%
```

---

## 7) What to say in your report

You can write something like this:

### Project extension
This work extends the baseline soccer video analysis pipeline by adding:
1. player pitch position estimation,
2. ball pitch position estimation,
3. heuristic team formation detection,
4. formation confidence scoring.

### Method
Object tracking is first performed using YOLO and ByteTrack. Then player and ball positions are projected into field coordinates using perspective transformation. Team membership is estimated from jersey color clustering. Finally, team formations are inferred by grouping outfield players into tactical lines and matching those line counts against common soccer formation templates.

### Limitation
The current approach estimates formation and confidence, but it does not produce a true predictive win-rate model. A win-rate model would require supervised learning with historical match-level labels.

---

## 8) If your professor insists on “training a model”

Then the next correct version is this:

### Option 1 – formation classifier
Train a model using features like:
- average player x positions
- average player y positions
- distances between lines
- width of formation
- compactness

Labels:
- `4-4-2`
- `4-3-3`
- `4-2-3-1`
- etc.

### Option 2 – win rate model
Train on historical match data with labels:
- formation
- possession
- shots
- xG
- passes
- defensive line height
- match result

Target:
- win / draw / loss

---

## 9) Best sentence to tell your professor

Use this exact wording if you want:

> “I implemented player and ball position estimation and added formation recognition with confidence scoring. Because win-rate prediction requires labeled historical match outcomes, I treated that part as a future supervised-learning extension rather than making an unsupported claim from a single video.”

