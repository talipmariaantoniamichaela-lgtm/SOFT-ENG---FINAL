"""
Generates synthetic and realistic test video files for validation of Motion Reader.
Renders humanoid actions: entering frame, standing, raising right hand, waving,
lowering hand, stepping forward, sitting down, and standing up.
"""
import os
import math
import cv2
import numpy as np


def generate_validation_video(output_path: str = "tests/samples/demo_motion_analysis.mp4", fps: int = 30, duration_sec: int = 22):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))

    total_frames = fps * duration_sec
    print(f"Generating validation video: {output_path} ({total_frames} frames)...")

    # Colors
    bg_color = (230, 235, 240)       # Light clean studio background
    skin_color = (190, 215, 240)     # Skin tone (BGR)
    hair_color = (30, 30, 35)        # Dark hair
    shirt_color = (40, 40, 45)       # Dark t-shirt matching reference image
    pants_color = (70, 50, 40)       # Dark navy jeans

    for f_idx in range(total_frames):
        t = f_idx / fps
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # Base center coordinates
        base_x = width // 2
        base_y = int(height * 0.72)
        scale = 1.0

        # Action timeline:
        # 0.0 - 1.5s: Standing neutral
        # 1.5 - 3.5s: Right hand raises
        # 3.5 - 6.5s: Right hand waves
        # 6.5 - 8.5s: Right hand lowers
        # 8.5 - 11.0s: Standing
        # 11.0 - 14.0s: Stepping forward (scale increases by 15%)
        # 14.0 - 17.0s: Stepping backward (scale returns)
        # 17.0 - 20.0s: Sitting down
        # 20.0 - 22.0s: Standing back up

        r_wrist_raised = False
        r_wrist_x = 0
        r_wrist_y = 0
        is_sitting = False

        if 11.0 <= t < 14.0:
            # Scale up (stepping forward)
            progress = (t - 11.0) / 3.0
            scale = 1.0 + 0.16 * math.sin(progress * math.pi / 2.0)
        elif 14.0 <= t < 17.0:
            # Scale down (stepping backward)
            progress = (t - 14.0) / 3.0
            scale = 1.16 - 0.16 * math.sin(progress * math.pi / 2.0)
        elif 17.0 <= t < 20.0:
            # Sitting down
            is_sitting = True
        elif t >= 20.0:
            # Standing back up
            is_sitting = False

        # Compute body heights scaled
        head_radius = int(32 * scale)
        head_cy = int((height * 0.25 if not is_sitting else height * 0.38) * scale)
        torso_top = head_cy + head_radius
        torso_bottom = int(torso_top + 130 * scale)
        shoulder_w = int(120 * scale)

        # Draw Head and Hair
        cv2.circle(frame, (base_x, head_cy), head_radius, skin_color, -1)
        # Hair
        cv2.ellipse(frame, (base_x, head_cy - int(8 * scale)), (head_radius, int(head_radius * 0.7)), 0, 180, 360, hair_color, -1)

        # Draw Eyes and Mouth
        eye_y = head_cy - int(2 * scale)
        cv2.circle(frame, (base_x - int(10 * scale), eye_y), int(3 * scale), (40, 40, 40), -1)
        cv2.circle(frame, (base_x + int(10 * scale), eye_y), int(3 * scale), (40, 40, 40), -1)
        cv2.ellipse(frame, (base_x, head_cy + int(12 * scale)), (int(8 * scale), int(4 * scale)), 0, 0, 180, (40, 40, 40), 2)

        # Draw Torso (Shirt)
        sh_left = (base_x - shoulder_w // 2, torso_top + int(15 * scale))
        sh_right = (base_x + shoulder_w // 2, torso_top + int(15 * scale))
        hip_left = (base_x - int(45 * scale), torso_bottom)
        hip_right = (base_x + int(45 * scale), torso_bottom)

        torso_pts = np.array([sh_left, sh_right, hip_right, hip_left], np.int32)
        cv2.fillPoly(frame, [torso_pts], shirt_color)

        # Draw Left Arm (hanging at side)
        l_elbow = (sh_left[0] - int(15 * scale), sh_left[1] + int(60 * scale))
        l_wrist = (sh_left[0] - int(10 * scale), sh_left[1] + int(115 * scale))
        cv2.line(frame, sh_left, l_elbow, skin_color, int(18 * scale))
        cv2.line(frame, l_elbow, l_wrist, skin_color, int(16 * scale))
        cv2.circle(frame, l_wrist, int(10 * scale), skin_color, -1)

        # Draw Right Arm (dependent on waving / raise)
        if 1.5 <= t < 3.5:
            # Raising right arm
            raise_prog = (t - 1.5) / 2.0
            r_elbow = (sh_right[0] + int(25 * scale), sh_right[1] + int(40 * (1 - raise_prog) * scale))
            r_wrist = (sh_right[0] + int(35 * scale), sh_right[1] - int(50 * raise_prog * scale))
        elif 3.5 <= t < 6.5:
            # Waving right arm
            wave_time = t - 3.5
            wave_dx = int(28 * math.sin(wave_time * 6.5) * scale)
            r_elbow = (sh_right[0] + int(30 * scale), sh_right[1] - int(10 * scale))
            r_wrist = (sh_right[0] + int(40 * scale) + wave_dx, sh_right[1] - int(65 * scale))
        elif 6.5 <= t < 8.5:
            # Lowering right arm
            lower_prog = (t - 6.5) / 2.0
            r_elbow = (sh_right[0] + int(25 * scale), sh_right[1] + int(40 * lower_prog * scale))
            r_wrist = (sh_right[0] + int(35 * scale), sh_right[1] - int(50 * (1 - lower_prog) * scale))
        else:
            # Normal down
            r_elbow = (sh_right[0] + int(15 * scale), sh_right[1] + int(60 * scale))
            r_wrist = (sh_right[0] + int(10 * scale), sh_right[1] + int(115 * scale))

        cv2.line(frame, sh_right, r_elbow, skin_color, int(18 * scale))
        cv2.line(frame, r_elbow, r_wrist, skin_color, int(16 * scale))
        cv2.circle(frame, r_wrist, int(10 * scale), skin_color, -1)

        # Draw Legs (Pants)
        if not is_sitting:
            # Standing legs
            knee_l = (hip_left[0] - int(5 * scale), hip_left[1] + int(85 * scale))
            knee_r = (hip_right[0] + int(5 * scale), hip_right[1] + int(85 * scale))
            ank_l = (knee_l[0], knee_l[1] + int(85 * scale))
            ank_r = (knee_r[0], knee_r[1] + int(85 * scale))
        else:
            # Sitting legs: knees bent
            knee_l = (hip_left[0] - int(30 * scale), hip_left[1] + int(35 * scale))
            knee_r = (hip_right[0] + int(30 * scale), hip_right[1] + int(35 * scale))
            ank_l = (knee_l[0], knee_l[1] + int(70 * scale))
            ank_r = (knee_r[0], knee_r[1] + int(70 * scale))

        cv2.line(frame, hip_left, knee_l, pants_color, int(22 * scale))
        cv2.line(frame, knee_l, ank_l, pants_color, int(18 * scale))
        cv2.line(frame, hip_right, knee_r, pants_color, int(22 * scale))
        cv2.line(frame, knee_r, ank_r, pants_color, int(18 * scale))

        # Shoes
        cv2.circle(frame, ank_l, int(12 * scale), (20, 20, 20), -1)
        cv2.circle(frame, ank_r, int(12 * scale), (20, 20, 20), -1)

        writer.write(frame)

    writer.release()
    print(f"Validation video created successfully at {output_path} ({os.path.getsize(output_path)} bytes)")


if __name__ == "__main__":
    generate_validation_video()
