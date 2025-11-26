import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
import math

class MinimalGestureRecognizer:
    def __init__(self):
        # MediaPipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        
        # Smoothing
        self.gesture_history = deque(maxlen=12)
        self.confidence_smooth = {"like": 0, "dislike": 0}
        
    def analyze_gesture(self, landmarks):
        """Analyze hand gesture with precision"""
        
        # Key points
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]
        
        index_tip, index_mcp = landmarks[8], landmarks[5]
        middle_tip, middle_mcp = landmarks[12], landmarks[9]
        ring_tip, ring_mcp = landmarks[16], landmarks[13]
        pinky_tip, pinky_mcp = landmarks[20], landmarks[17]
        
        # Check fingers closed
        fingers_closed = sum([
            index_tip.y > index_mcp.y,
            middle_tip.y > middle_mcp.y,
            ring_tip.y > ring_mcp.y,
            pinky_tip.y > pinky_mcp.y
        ])
        
        # Thumb direction
        thumb_vector_y = thumb_tip.y - thumb_mcp.y
        thumb_vector_x = thumb_tip.x - thumb_mcp.x
        
        # Thumb angle from horizontal
        angle = math.degrees(math.atan2(thumb_vector_y, abs(thumb_vector_x)))
        
        # Thumb extension check
        thumb_dist = math.sqrt((thumb_tip.x - wrist.x)**2 + (thumb_tip.y - wrist.y)**2)
        mcp_dist = math.sqrt((thumb_mcp.x - wrist.x)**2 + (thumb_mcp.y - wrist.y)**2)
        thumb_extended = thumb_dist > mcp_dist * 1.1
        
        like_score = 0
        dislike_score = 0
        
        if fingers_closed >= 3 and thumb_extended:
            # LIKE: thumb up
            if thumb_vector_y < -0.05:
                like_score += 35
            if angle < -25:
                like_score += 35
            if thumb_tip.y < thumb_mcp.y:
                like_score += 20
            if thumb_tip.y < index_mcp.y:
                like_score += 10
                
            # DISLIKE: thumb down
            if thumb_vector_y > 0.05:
                dislike_score += 35
            if angle > 25:
                dislike_score += 35
            if thumb_tip.y > thumb_mcp.y:
                dislike_score += 15
            if thumb_tip.y > wrist.y:
                dislike_score += 15
            if thumb_tip.y > middle_mcp.y:
                dislike_score += 10
        
        return min(100, like_score), min(100, dislike_score)
    
    def get_smoothed_gesture(self, gesture):
        """Smooth detection"""
        self.gesture_history.append(gesture)
        
        if len(self.gesture_history) < 5:
            return gesture
            
        counts = {"LIKE": 0, "DISLIKE": 0, "NONE": 0}
        for g in self.gesture_history:
            counts[g] += 1
        
        total = len(self.gesture_history)
        if counts["LIKE"] / total > 0.55:
            return "LIKE"
        elif counts["DISLIKE"] / total > 0.55:
            return "DISLIKE"
        return "NONE"
    
    def draw_minimal_ui(self, frame, gesture, like_conf, dislike_conf, hand_detected):
        """Clean minimal UI"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        
        # Top bar
        cv2.rectangle(overlay, (0, 0), (w, 70), (15, 15, 15), -1)
        
        # Bottom bar  
        cv2.rectangle(overlay, (0, h-50), (w, h), (15, 15, 15), -1)
        
        # Blend
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Title - minimal
        cv2.putText(frame, "GESTURE", (30, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Minimal confidence indicators
        self.draw_minimal_bar(frame, w-280, 25, like_conf, (0, 200, 120), "LIKE")
        self.draw_minimal_bar(frame, w-140, 25, dislike_conf, (80, 80, 220), "DISLIKE")
        
        # Center gesture display
        if gesture != "NONE":
            self.draw_gesture_display(frame, gesture, w, h)
        else:
            if hand_detected:
                cv2.putText(frame, "Analyzing...", (w//2 - 70, h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            else:
                # Minimal hand icon hint
                self.draw_hand_hint(frame, w//2, h//2)
        
        # Bottom instruction
        cv2.putText(frame, "Q: Quit", (30, h-18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
        
        return frame
    
    def draw_minimal_bar(self, frame, x, y, value, color, label):
        """Draw minimal progress bar"""
        bar_w, bar_h = 100, 6
        
        # Label
        cv2.putText(frame, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        
        # Background
        cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (40, 40, 40), -1)
        
        # Fill
        fill_w = int(bar_w * value / 100)
        if fill_w > 0:
            cv2.rectangle(frame, (x, y), (x + fill_w, y + bar_h), color, -1)
    
    def draw_gesture_display(self, frame, gesture, w, h):
        """Draw clean gesture result"""
        
        if gesture == "LIKE":
            color = (0, 200, 120)  # Green
            icon_points = self.get_thumb_icon(w//2, h//2 + 60, True)
        else:
            color = (80, 80, 220)  # Red
            icon_points = self.get_thumb_icon(w//2, h//2 + 60, False)
        
        # Gesture text
        text_size = cv2.getTextSize(gesture, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
        text_x = w//2 - text_size[0]//2
        
        # Subtle shadow
        cv2.putText(frame, gesture, (text_x + 2, h//2 - 20 + 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # Main text
        cv2.putText(frame, gesture, (text_x, h//2 - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
        
        # Minimal icon
        cv2.fillPoly(frame, [icon_points], color)
    
    def get_thumb_icon(self, cx, cy, is_up):
        """Get minimal thumb icon points"""
        if is_up:
            points = np.array([
                [cx, cy - 25],
                [cx - 8, cy - 10],
                [cx - 8, cy + 15],
                [cx + 8, cy + 15],
                [cx + 8, cy - 10]
            ], np.int32)
        else:
            points = np.array([
                [cx, cy + 25],
                [cx - 8, cy + 10],
                [cx - 8, cy - 15],
                [cx + 8, cy - 15],
                [cx + 8, cy + 10]
            ], np.int32)
        return points
    
    def draw_hand_hint(self, frame, cx, cy):
        """Draw minimal hand hint"""
        cv2.putText(frame, "Show hand gesture", (cx - 100, cy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
        
        # Simple circle hint
        cv2.circle(frame, (cx, cy + 40), 30, (50, 50, 50), 1)
        cv2.circle(frame, (cx, cy + 40), 5, (50, 50, 50), -1)
    
    def draw_minimal_hand(self, frame, landmarks, w, h):
        """Draw very minimal hand outline"""
        
        # Only draw essential connections - palm outline
        palm_indices = [0, 1, 5, 9, 13, 17, 0]
        
        points = []
        for idx in palm_indices:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            points.append([x, y])
        
        points = np.array(points, np.int32)
        cv2.polylines(frame, [points], False, (60, 60, 60), 1)
        
        # Finger lines - very subtle
        fingers = [
            [1, 2, 3, 4],    # Thumb
            [5, 6, 7, 8],    # Index
            [9, 10, 11, 12], # Middle
            [13, 14, 15, 16],# Ring
            [17, 18, 19, 20] # Pinky
        ]
        
        for finger in fingers:
            for i in range(len(finger) - 1):
                p1 = (int(landmarks[finger[i]].x * w), int(landmarks[finger[i]].y * h))
                p2 = (int(landmarks[finger[i+1]].x * w), int(landmarks[finger[i+1]].y * h))
                cv2.line(frame, p1, p2, (50, 50, 50), 1)
        
        # Only thumb tip indicator
        thumb_tip = landmarks[4]
        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
        cv2.circle(frame, (tx, ty), 4, (100, 100, 100), -1)
        
        return frame
    
    def run(self):
        """Main loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n" + "─" * 40)
        print("  MINIMAL GESTURE RECOGNITION")
        print("─" * 40)
        print("  👍 Thumbs Up  → LIKE")
        print("  👎 Thumbs Down → DISLIKE")
        print("  Q → Quit")
        print("─" * 40 + "\n")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Process
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            gesture = "NONE"
            like_conf = 0
            dislike_conf = 0
            hand_detected = False
            
            if results.multi_hand_landmarks:
                hand_detected = True
                landmarks = results.multi_hand_landmarks[0].landmark
                
                # Draw minimal hand
                frame = self.draw_minimal_hand(frame, landmarks, w, h)
                
                # Analyze
                like_conf, dislike_conf = self.analyze_gesture(landmarks)
                
                # Smooth confidence
                self.confidence_smooth["like"] = self.confidence_smooth["like"] * 0.6 + like_conf * 0.4
                self.confidence_smooth["dislike"] = self.confidence_smooth["dislike"] * 0.6 + dislike_conf * 0.4
                
                # Determine gesture
                if like_conf >= 70:
                    gesture = "LIKE"
                elif dislike_conf >= 70:
                    gesture = "DISLIKE"
            else:
                # Fade out confidence when no hand
                self.confidence_smooth["like"] *= 0.9
                self.confidence_smooth["dislike"] *= 0.9
            
            # Smooth gesture
            final_gesture = self.get_smoothed_gesture(gesture)
            
            # Draw UI
            frame = self.draw_minimal_ui(
                frame, 
                final_gesture,
                self.confidence_smooth["like"],
                self.confidence_smooth["dislike"],
                hand_detected
            )
            
            cv2.imshow("Gesture", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()


if __name__ == "__main__":
    app = MinimalGestureRecognizer()
    app.run()
