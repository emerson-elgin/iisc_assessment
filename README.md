# README: Baby Movement Analysis using Pose Estimation

```
## Project Overview
This project analyzes baby movement patterns using a pose estimation model (MediaPipe Pose). The extracted keypoints are used to compute four key movement features, which help in understanding mobility patterns and detecting anomalies.

## Features Extracted
1. **Histogram of Angular Displacement (HOAD2D):**
   - Measures the change in joint angles over time (Δt = 10 frames).
   - A wider spread indicates dynamic movement, while a peak near zero suggests limited motion.

2. **Histogram of Relative Joint Orientation (HORJO2D):**
   - Captures how joints are oriented relative to each other.
   - Helps analyze posture consistency and coordination.

3. **Histogram of Joint Orientation (HOJO2D):**
   - Measures the absolute orientation of joints.
   - Peaks suggest frequent postures, while a broad distribution indicates continuous movement.

4. **Histogram of Joint Displacement (HOJD2D):**
   - Computes the Euclidean distance each joint moves between frames.
   - Higher displacement values indicate active motion, while lower values suggest stillness.

## Project Workflow
1. **Pose Estimation:**
   - The system processes a baby movement video and extracts 2D joint coordinates using MediaPipe Pose.
   
2. **Feature Computation:**
   - The extracted keypoints are analyzed to compute HOAD2D, HORJO2D, HOJO2D, and HOJD2D.
   
3. **Data Visualization:**
   - Histograms of each feature are plotted to observe movement trends.
   
4. **Interpretation:**
   - The histograms help in identifying normal vs abnormal movement patterns.

## Installation & Requirements
Ensure you have the required dependencies installed before running the project.

```bash
pip install opencv-python mediapipe numpy matplotlib
```

## Running the Code
1. Place your baby movement video in the project folder and update the filename in the script.
2. Run the main script:
   ```bash
   python pose_analysis.py
   ```
3. The system will generate:
   - A **skeleton overlay video** (`skeleton_output.mp4`).
   - **Histograms of movement features** for analysis.

## Output Interpretation
- If histograms show a **wide spread**, the baby exhibits diverse movements.
- A **narrow peak** in displacement or orientation suggests minimal motion, possibly indicating movement restrictions.
- Repetitive peaks in joint orientation may suggest stereotyped movement patterns.

## Applications
- Early detection of **cerebral palsy** or other motor impairments.
- Assessing **movement variability** in infants.
- Studying **postural coordination** during development.

## Contributors
- **Project Developer:** Emerson
- **Reference Paper:**  Cerebral palsy detection from infant using movements 
of their salient body parts and a feature fusion model(https://doi.org/10.1007/s11227-024-06520-z)

## License
This project is open-source and can be modified or extended for research purposes.

---
For any issues or contributions, feel free to reach out!
```

