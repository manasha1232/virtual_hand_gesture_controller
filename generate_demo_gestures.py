import os
import cv2
import numpy as np

def draw_synthetic_hand(img, center_x, center_y, gesture_name="open_palm", scale=1.0):
    """Draws a synthetic hand skeleton representation for automated benchmarking."""
    # Hand palm color
    skin_color = (180, 210, 240)
    joint_color = (50, 50, 220)
    bone_color = (0, 255, 0)
    
    wrist = (int(center_x), int(center_y + 120 * scale))
    cv2.circle(img, wrist, int(12 * scale), joint_color, -1)
    
    # Palm Base
    palm_center = (int(center_x), int(center_y + 40 * scale))
    cv2.ellipse(img, palm_center, (int(45 * scale), int(55 * scale)), 0, 0, 360, skin_color, -1)
    cv2.ellipse(img, palm_center, (int(45 * scale), int(55 * scale)), 0, 0, 360, (100, 100, 100), 2)
    
    # Finger bases (MCP joints)
    mcp_x = [center_x - 35 * scale, center_x - 12 * scale, center_x + 12 * scale, center_x + 35 * scale]
    mcp_y = [center_y, center_y - 10 * scale, center_y - 8 * scale, center_y + 5 * scale]
    
    # Finger states based on gesture
    # Extensions: [Thumb, Index, Middle, Ring, Pinky] (True = UP, False = CLOSED)
    if gesture_name == "open_palm":
        extensions = [True, True, True, True, True]
    elif gesture_name == "fist":
        extensions = [False, False, False, False, False]
    elif gesture_name == "peace":
        extensions = [False, True, True, False, False]
    elif gesture_name == "pointing":
        extensions = [False, True, False, False, False]
    elif gesture_name == "pinch":
        extensions = [True, True, False, False, False] # Pinch brings thumb & index tip together
    elif gesture_name == "thumbs_up":
        extensions = [True, False, False, False, False]
    else:
        extensions = [True, True, True, True, True]
        
    finger_lengths = [70 * scale, 95 * scale, 105 * scale, 90 * scale, 75 * scale]
    
    # Draw 4 fingers (Index, Middle, Ring, Pinky)
    for i in range(4):
        bx, by = int(mcp_x[i]), int(mcp_y[i])
        ext = extensions[i + 1]
        flen = finger_lengths[i + 1]
        
        cv2.line(img, wrist, (bx, by), bone_color, 3)
        cv2.circle(img, (bx, by), int(8 * scale), joint_color, -1)
        
        if ext: # Extended Finger
            tip_x = bx + (i - 1.5) * 5 * scale
            tip_y = by - flen
            mid_x = (bx + tip_x) / 2.0
            mid_y = (by + tip_y) / 2.0
        else: # Folded Finger (Fist)
            mid_x = bx
            mid_y = by + 25 * scale
            tip_x = bx
            tip_y = by + 45 * scale
            
        mid_pt = (int(mid_x), int(mid_y))
        tip_pt = (int(tip_x), int(tip_y))
        
        cv2.line(img, (bx, by), mid_pt, bone_color, 3)
        cv2.line(img, mid_pt, tip_pt, bone_color, 3)
        cv2.circle(img, mid_pt, int(6 * scale), joint_color, -1)
        cv2.circle(img, tip_pt, int(8 * scale), (0, 0, 255), -1) # Red fingertip landmark
        
    # Draw Thumb
    thumb_mcp = (int(center_x - 40 * scale), int(center_y + 40 * scale))
    cv2.line(img, wrist, thumb_mcp, bone_color, 3)
    cv2.circle(img, thumb_mcp, int(8 * scale), joint_color, -1)
    
    if gesture_name == "pinch":
        # Thumb tip touches index tip
        index_tip = (int(mcp_x[0]), int(mcp_y[0] - finger_lengths[1]))
        thumb_tip = index_tip
    elif gesture_name == "thumbs_up":
        thumb_tip = (int(center_x - 45 * scale), int(center_y - 80 * scale))
    elif extensions[0]: # Extended Thumb
        thumb_tip = (int(center_x - 75 * scale), int(center_y - 20 * scale))
    else: # Folded Thumb
        thumb_tip = (int(center_x - 20 * scale), int(center_y + 30 * scale))
        
    cv2.line(img, thumb_mcp, thumb_tip, bone_color, 3)
    cv2.circle(img, thumb_tip, int(8 * scale), (0, 0, 255), -1)

def generate_synthetic_gestures_video(output_path="input/sample_hand_gestures.mp4", width=800, height=600, fps=30):
    """Generates a synthetic 180-frame gesture sequence video."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        output_path = output_path.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
    print(f"[+] Generating synthetic hand gesture video: '{output_path}'...")
    
    gestures_sequence = [
        ("open_palm", 30),
        ("fist", 30),
        ("peace", 30),
        ("pointing", 30),
        ("pinch", 30),
        ("thumbs_up", 30)
    ]
    
    frame_idx = 0
    bg = np.ones((height, width, 3), dtype=np.uint8) * 30
    # Add ambient grid pattern
    for x in range(0, width, 40):
        cv2.line(bg, (x, 0), (x, height), (45, 45, 45), 1)
    for y in range(0, height, 40):
        cv2.line(bg, (0, y), (width, y), (45, 45, 45), 1)
        
    for gest, duration in gestures_sequence:
        for f in range(duration):
            frame_idx += 1
            frame = bg.copy()
            
            # Oscillate hand position slightly to simulate human micro-motion
            offset_x = int(15 * np.sin(f / 5.0))
            offset_y = int(10 * np.cos(f / 5.0))
            
            cx = width // 2 + offset_x
            cy = height // 2 + offset_y
            
            draw_synthetic_hand(frame, cx, cy, gesture_name=gest, scale=1.1)
            
            # Frame overlay text
            cv2.putText(frame, f"SYNTHETIC GESTURE SEQUENCE: {gest.upper()}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)
            cv2.putText(frame, f"FRAME: {frame_idx:03d} / 180", (width - 200, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        
            out.write(frame)
            
    out.release()
    print(f"[OK] Synthetic gesture video created at '{output_path}'")
    return output_path

if __name__ == "__main__":
    generate_synthetic_gestures_video()
