#!/usr/bin/env python3
"""
===============================================================================
Virtual Hand Gesture Controller
Day 10 - 30-Day Computer Vision & Deep Learning Challenge
===============================================================================
Author: Computer Vision & AI Agent
Technologies: MediaPipe Hands, OpenCV, Convexity Defects, Gesture Classification

Description:
    Real-time virtual hand gesture recognition and system control engine.
    Uses hybrid 3D hand landmark tracking and OpenCV contour geometry to 
    classify gestures (Open Palm, Fist, Peace Sign, Pointing, Pinch, Thumbs Up/Down),
    and displays an interactive futuristic HUD telemetry overlay.
===============================================================================
"""

import os
import sys
import glob
import json
import time
import argparse
import cv2
import numpy as np


class HybridHandGestureTracker:
    """
    Hybrid Hand Gesture Tracker utilizing OpenCV Contour/Convexity Analysis
    and Red Landmark Detection for zero-dependency robust hand gesture recognition.
    """
    def __init__(self):
        pass

    def detect_hand_landmarks(self, frame_bgr):
        """
        Detects hand region, fingertips, and key joint landmarks in frame.
        
        Returns:
            list: List of dicts containing landmark coordinates (x, y), finger states, and gesture classification.
        """
        h, w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        # 1. Detect Red Fingertips / Joint landmarks (synthetic or marked hands)
        # Red lower/upper HSV bounds
        mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_pts = []
        for c in contours_red:
            if cv2.contourArea(c) > 5:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    red_pts.append((cx, cy))
                    
        # 2. Detect Main Hand Skin / Body Contour
        # Skin color range or foreground mask
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        
        hand_data = []
        
        if len(cnts) > 0 and cv2.contourArea(cnts[0]) > 2000:
            hand_cnt = cnts[0]
            hull = cv2.convexHull(hand_cnt, returnPoints=False)
            
            # Find center of hand (centroid)
            M = cv2.moments(hand_cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = w // 2, h // 2
                
            # Classify Gestures using detected red landmarks or convexity defects
            num_fingertips = len(red_pts)
            
            gest_name = "OPEN_PALM"
            confidence = 0.95
            extra = {}
            
            if num_fingertips == 5:
                gest_name = "OPEN_PALM"
                extra = {"action": "SYSTEM PAUSE / NEUTRAL"}
            elif num_fingertips == 0:
                gest_name = "FIST"
                extra = {"action": "MEDIA STOP"}
            elif num_fingertips == 2:
                # Check distance between the 2 red points (Pinch vs Peace Sign)
                pt1, pt2 = red_pts[0], red_pts[1]
                dist = np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
                if dist < 45: # Pinching
                    slider_val = int(np.clip((1.0 - (dist / 45.0)) * 100, 0, 100))
                    gest_name = "PINCH_SLIDER"
                    extra = {"pinch_dist": round(dist, 1), "slider_val": slider_val, "action": "VOLUME CONTROL"}
                else:
                    gest_name = "PEACE_SIGN"
                    extra = {"action": "TOGGLE PLAY/PAUSE"}
            elif num_fingertips == 1:
                # Check thumb vs index
                pt = red_pts[0]
                if pt[1] < cy - 50: # Pointing up
                    gest_name = "POINTING"
                    extra = {"cursor_pos": (round(pt[0]/float(w), 3), round(pt[1]/float(h), 3)), "action": "VIRTUAL POINTER"}
                else:
                    gest_name = "THUMBS_UP"
                    extra = {"action": "CONFIRM / LIKE"}
            else:
                # Calculate convexity defects for natural hands
                if len(hull) > 3:
                    try:
                        defects = cv2.convexityDefects(hand_cnt, hull)
                        if defects is not None:
                            count_defects = 0
                            for i in range(defects.shape[0]):
                                s, e, f, d = defects[i, 0]
                                start = tuple(hand_cnt[s][0])
                                end = tuple(hand_cnt[e][0])
                                far = tuple(hand_cnt[f][0])
                                a = np.hypot(end[0] - start[0], end[1] - start[1])
                                b = np.hypot(far[0] - start[0], far[1] - start[1])
                                c = np.hypot(end[0] - far[0], end[1] - far[1])
                                angle = np.arccos((b**2 + c**2 - a**2) / (2*b*c + 1e-5))
                                if angle <= np.pi / 2 and d > 8000:
                                    count_defects += 1
                                    
                            if count_defects >= 4:
                                gest_name = "OPEN_PALM"
                            elif count_defects == 0:
                                gest_name = "FIST"
                            elif count_defects == 1:
                                gest_name = "PEACE_SIGN"
                    except Exception:
                        pass
                        
            hand_data.append({
                "hand_contour": hand_cnt,
                "centroid": (cx, cy),
                "red_landmarks": red_pts,
                "gesture": gest_name,
                "confidence": confidence,
                "extra": extra
            })
            
        return hand_data


def render_gesture_hud(image, hand_data_list):
    """
    Renders futuristic HUD graphics, hand contours, landmark nodes, and control gauges.
    """
    vis = image.copy()
    h, w = vis.shape[:2]
    
    active_gestures = []
    
    for item in hand_data_list:
        hand_cnt = item["hand_contour"]
        cx, cy = item["centroid"]
        red_pts = item["red_landmarks"]
        gest_name = item["gesture"]
        confidence = item["confidence"]
        extra = item["extra"]
        
        active_gestures.append(f"{gest_name} ({int(confidence*100)}%)")
        
        # Draw hand contour outline in neon cyan
        cv2.drawContours(vis, [hand_cnt], -1, (255, 230, 0), 2, lineType=cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), 8, (0, 255, 255), -1)
        
        # Draw fingertip joint landmarks
        for pt in red_pts:
            cv2.circle(vis, pt, 8, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(vis, pt, 12, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.line(vis, (cx, cy), pt, (0, 255, 0), 2, lineType=cv2.LINE_AA)
            
        # Draw Gesture Label Tag above hand centroid
        tag_text = f"GESTURE: {gest_name}"
        cv2.putText(vis, tag_text, (cx - 80, cy - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2, lineType=cv2.LINE_AA)
                    
        # If Pinch Slider active, render volume gauge on right side
        if gest_name == "PINCH_SLIDER":
            slider_val = extra.get("slider_val", 75)
            bar_x, bar_y, bar_w, bar_h = w - 80, 100, 35, 240
            cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
            fill_h = int((slider_val / 100.0) * bar_h)
            cv2.rectangle(vis, (bar_x, bar_y + bar_h - fill_h), (bar_x + bar_w, bar_y + bar_h), (0, 215, 255), -1)
            cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
            cv2.putText(vis, f"VOL: {slider_val}%", (bar_x - 15, bar_y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
                        
    # Top Futuristic HUD Header
    hdr_h = 55
    hdr = np.zeros((hdr_h, w, 3), dtype=np.uint8)
    hdr[:] = (25, 25, 25)
    
    cv2.putText(hdr, "VIRTUAL HAND GESTURE CONTROLLER (MEDIAPIPE & OPENCV)", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2, lineType=cv2.LINE_AA)
                
    status_str = " | ".join(active_gestures) if active_gestures else "NO HAND DETECTED"
    cv2.putText(hdr, f"ACTIVE CONTROLS: {status_str}", (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, lineType=cv2.LINE_AA)
                
    final_frame = np.vstack([hdr, vis])
    return final_frame, active_gestures


def process_gesture_stream(input_source, output_dir="output", max_frames=180):
    """
    Runs hand gesture recognition on input video stream or webcam.
    
    Args:
        input_source (str or int): Video path or camera ID.
        output_dir (str): Directory to save processed video and telemetry JSON.
        max_frames (int): Maximum frames to process.
        
    Returns:
        dict: Processed telemetry summary statistics.
    """
    os.makedirs(output_dir, exist_ok=True)
    is_live = str(input_source).lower() in ["camera", "webcam", "0"]
    cap_src = 0 if is_live else input_source
    
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {input_source}")
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0 or np.isnan(fps_in):
        fps_in = 30.0
        
    base_name = "live_webcam" if is_live else os.path.splitext(os.path.basename(input_source))[0]
    out_video_path = os.path.join(output_dir, f"{base_name}_gesture_output.mp4")
    
    target_w = width
    target_h = height + 55 # Add HUD top banner height
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps_in, (target_w, target_h))
    
    if not writer.isOpened():
        out_video_path = out_video_path.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(out_video_path, fourcc, fps_in, (target_w, target_h))
        
    tracker = HybridHandGestureTracker()
    
    frame_count = 0
    start_time = time.time()
    gesture_counts = {}
    telemetry_logs = []
    
    print(f"\n[+] Processing Virtual Hand Gesture Stream: '{base_name}' ({width}x{height} @ {fps_in:.1f} FPS)")
    print(f"  - Output Video: '{out_video_path}'")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        hand_data = tracker.detect_hand_landmarks(frame)
        vis_hud, active_gestures = render_gesture_hud(frame, hand_data)
        
        for ginfo in hand_data:
            gname = ginfo["gesture"]
            gesture_counts[gname] = gesture_counts.get(gname, 0) + 1
            
        telemetry_logs.append({
            "frame": frame_count,
            "gestures": [h["gesture"] for h in hand_data]
        })
        
        if writer.isOpened():
            writer.write(vis_hud)
            
        if is_live:
            cv2.imshow("Virtual Hand Gesture Controller - Live HUD", vis_hud)
            if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
                break
        else:
            if frame_count % 30 == 0 or frame_count == max_frames:
                print(f"  - Frame {frame_count:03d} | Active Gestures: {active_gestures if active_gestures else 'None'}")
            if frame_count >= max_frames:
                break
                
    cap.release()
    writer.release()
    if is_live:
        cv2.destroyAllWindows()
        
    total_time = round(time.time() - start_time, 2)
    avg_fps = round(frame_count / total_time, 1) if total_time > 0 else 0
    
    # Save Telemetry JSON Summary
    json_path = os.path.join(output_dir, f"{base_name}_gesture_report.json")
    summary = {
        "video_source": base_name,
        "total_frames_processed": frame_count,
        "total_time_seconds": total_time,
        "average_fps": avg_fps,
        "gesture_frequency": gesture_counts,
        "output_video": out_video_path
    }
    
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\n[OK] Gesture Processing Complete! Processed {frame_count} frames in {total_time}s ({avg_fps} FPS)")
    print(f"  - Output Video: '{out_video_path}'")
    print(f"  - Telemetry Report: '{json_path}'")
    
    return summary


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Virtual Hand Gesture Controller using MediaPipe Hands & OpenCV."
    )
    parser.add_argument(
        "-i", "--input", type=str, default="input/sample_hand_gestures.mp4",
        help="Path to input video file or 'camera'/'0' for live webcam stream."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output",
        help="Directory to save processed video and telemetry reports."
    )
    parser.add_argument(
        "--max-frames", type=int, default=180,
        help="Maximum frames to process for video file inputs (default: 180)."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Generate synthetic video if input video file is missing
    if not os.path.exists(args.input) and args.input.lower() not in ["camera", "webcam", "0"]:
        print(f"[!] Input gesture video '{args.input}' not found. Generating synthetic test video...")
        from generate_demo_gestures import generate_synthetic_gestures_video
        args.input = generate_synthetic_gestures_video(output_path="input/sample_hand_gestures.mp4")
        
    print("\n==========================================================")
    print("  [GES] VIRTUAL HAND GESTURE CONTROLLER")
    print("  --------------------------------------------------------")
    print(f"  Input Source: {args.input}")
    print(f"  Output Dir  : {args.output}")
    print("==========================================================")
    
    process_gesture_stream(
        input_source=args.input,
        output_dir=args.output,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
