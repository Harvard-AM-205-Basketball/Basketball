# Citation: https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

import numpy as np
import cv2
import sys
 
(major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')
 
def bbox_corners(bbox):
    """Get the coordinates of a bounding box"""
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    return p1, p2


def calc_fg1(frame, frame_bg):
    """Calculate a foreground frame given a frame and a background."""
    x = (frame.astype(np.int16) - frame_bg.astype(np.int16)) + 32
    return np.clip(x, 0, 255).astype(np.uint8)


def calc_fg2(frame, fgmask):
    return cv2.bitwise_and(frame, frame, mask=fgmask)


if __name__ == '__main__' :
 
    # Set up tracker.
    # Instead of MIL, you can also use
 
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    # tracker_type = tracker_types[2]
    tracker_type = 'CSRT'
    
    print(f'Attempting object tracking with {tracker_type}...')
 
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
 
    # Background frame
    frame_bg = cv2.imread('../frames/Background/Camera3_median.png')

    # Read video
    video = cv2.VideoCapture("../sync_video/Camera3_sync.mp4")

    # Background subtractor
    fgbg = cv2.createBackgroundSubtractorMOG2()
 
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    
    # Get the foreground
    # frame_fg = calc_fg(frame, frame_bg)
    fgmask = fgbg.apply(frame)
    frame_fg = calc_fg2(frame, fgmask)
    
    # Table of interactive bounding boxes; key = frame number, value = bounding box
    bbox_interactive = dict()
        
    # Interactively select a bounding box
    bbox = cv2.selectROI(frame_fg, False)
    bbox_interactive[0] = bbox
 
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame_fg, bbox)
 
    # Get the number of frames
    # property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
    # frame_count: int = max(int(cv2.VideoCapture.get(video, property_id)), 3390)
    
    # Initialize array of bounding boxes
    bbox_tbl = np.zeros((4391, 4), dtype=np.int16)
    
    # Save this box to the table
    bbox_tbl[0, :] = np.array(bbox)
    
    # Count the frames
    fn: int = 0
    
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break         

        # Increment frame count (just read a new frame)
        fn += 1
    
        # Foreground
        # frame_fg = calc_fg(frame, frame_bg)
        fgmask = fgbg.apply(frame)
        frame_fg = calc_fg2(frame, fgmask)
    
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        ok, bbox = tracker.update(frame_fg)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
        # Draw bounding box
        if ok:
            # Tracking success
            p1, p2 = bbox_corners(bbox)
            # cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)            
            cv2.rectangle(frame_fg, p1, p2, (255,0,0), 2, 1)
            # MSE
            # Save this box to the table
            bbox_tbl[fn, :] = np.array(bbox)
            # Status update
            print(f'frame={fn}, p1={p1}, p2={p2}')
        else :
            # Tracking failure
            cv2.putText(frame_fg, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            if fn % 300 == 0:
                # Ask if ball is present on the console
                print(f'Tracking failed at frame {fn}.  Is the ball on the screen? (Y/N)')
                ball_input = input()
                ball_present: bool = (ball_input == 'y')
                # If this frame is a multiple of one second, try to pick a new box interactively
                if ball_present:
                    # Pick a new box interactively
                    bbox = cv2.selectROI(frame_fg, False)
                    # Save it to both the main and interactive tables
                    bbox_tbl[fn, :] = np.array(bbox)
                    bbox_interactive[fn] = bbox

        # Display tracker type on frame
        cv2.putText(frame_fg, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(frame_fg, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
        cv2.imshow("Tracking", frame_fg)
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
    
    # Print final frame count to console
    print(f'Total Frame Count: {fn}')
    