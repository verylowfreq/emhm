#!/usr/bin/env python

# MIT License
#
# Copyright (c) 2023 verylowfreq
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

PROGRAM_SHORT_DESCRIPTION = "emhm"
LICENSE_DESCRIPTION = "Licensed under MIT License (C) 2023 verylowfreq"


from typing import List, NoReturn, Optional, Any, Tuple, Iterable
from pprint import pprint
import traceback
import sys
import math
import time
import argparse
import os
import threading
from queue import Queue
import cv2
import numpy as np

import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles
import mediapipe.python.solutions.pose as mp_pose


class App:
    DEFAULT_DETECTION_THRESHOLD = 0.5
    DEFAULT_TRACKING_THRESHOLD = 0.5
    DEFAULT_MASK_THRESHOLD = 0.1
    DEFAULT_MASK_EXPANSION = 0
    DEFUALT_MODEL_COMPLEXITY = 1  # 0, 1, or 2 (Full)
    DEFAULT_BONE_WIDTH = 0.005
    DEFAULT_BONE_COLOR = "white"

    MAXWIDTH_FOR_MP = 1024
    MAXHEIGHT_FOR_MP = 1024

    class Finished(Exception):
        pass

    def __init__(self) -> None:
        print(PROGRAM_SHORT_DESCRIPTION + " " + LICENSE_DESCRIPTION)
        self.parse_args(sys.argv)


    def parse_args(self, args:List[str]) -> None:
        """
        Parse arguments given to this program via commandline.
        """
        self.argparser = argparse.ArgumentParser()
        self.argparser.add_argument('-i', '--input', required=True, help='Input file path')
        self.argparser.add_argument('-o', '--output', required=True, help='Output file path')
        self.argparser.add_argument('-m', '--mask-threshold', type=float, default=App.DEFAULT_MASK_THRESHOLD, help='Threshold for mask area (0.0-1.0)')
        self.argparser.add_argument('-e', '--mask-expansion', type=int, default=App.DEFAULT_MASK_EXPANSION, help='How extent mask area (1...)')
        self.argparser.add_argument('-d', '--detection-threshold', type=float, default=App.DEFAULT_DETECTION_THRESHOLD, help='Threshold for detection in estimation')
        self.argparser.add_argument('-t', '--tracking-threshold', type=float, default=App.DEFAULT_TRACKING_THRESHOLD, help='Threshold for tracking in estimation')
        self.argparser.add_argument('--draw-bones', action='store_true', help='For debug. Draw the detected bone lines')
        self.argparser.add_argument('--bone-width', type=float, default=App.DEFAULT_BONE_WIDTH, help=f'Width of bones in ratio to screen width (0.0-1.0; default {App.DEFAULT_BONE_WIDTH})')
        self.argparser.add_argument('--bone-color', choices=['white','black','red','green','blue'], default=App.DEFAULT_BONE_COLOR, help=f'Bone color (white,black,red,green,blue ; default {App.DEFAULT_BONE_COLOR})')
        self.argparser.add_argument('-c', '--model-complexity', choices='012', default=App.DEFUALT_MODEL_COMPLEXITY, help='ML model complexity(0, 1, or 2 (more accurate, heavy))')
        self.argparser.add_argument('-y', '--allow-overwrite', action='store_true', help='Overwrite the output file if exists')
        self.argparser.add_argument('--use-inpaint-foots', action='store_true', help='Use inpainting algorithm for foots')
        self.options = self.argparser.parse_args(args[1:])


    def draw_graysquareboxpattern(self, width:int, height:int) -> np.ndarray:
        """
        Generate a gray checkered pattern
        """
        BLOCK_SIZE = 32
        LIGHTGRAY = (150, 150, 150)
        DARKGRAY = (128, 128, 128)
        mat = np.zeros((height, width, 3), dtype=np.uint8)
        start_with_light_gray = True
        draw_light_gray = True
        for y in range(0, height, BLOCK_SIZE):
            draw_light_gray = start_with_light_gray
            for x in range(0, width, BLOCK_SIZE):
                if draw_light_gray:
                    mat[y:(y+BLOCK_SIZE), x:(x+BLOCK_SIZE)] = LIGHTGRAY
                else:
                    mat[y:(y+BLOCK_SIZE), x:(x+BLOCK_SIZE)] = DARKGRAY
                draw_light_gray = not draw_light_gray
            start_with_light_gray = not start_with_light_gray
        return mat


    @staticmethod
    def get_center_of_points(*points) -> Tuple[float, float]:
        x_sum = sum([p.x for p in points])
        y_sum = sum([p.y for p in points])
        x = x_sum / len(points)
        y = y_sum / len(points)
        return (x, y)

    @staticmethod
    def get_distance(pos1, pos2) -> float:
        x_distance = math.fabs(pos1.x - pos2.x)
        y_distance = math.fabs(pos1.y - pos2.y)
        return math.sqrt(x_distance ** 2 + y_distance ** 2)
    
    @staticmethod
    def get_center_and_radious_for_foot(width:int, height:int, landmarks:Iterable, ankle, heel, footindex) -> Tuple[Tuple[int, int], int]:
        ankle = landmarks[ankle]
        heel = landmarks[heel]
        footindex = landmarks[footindex]
        center_pos = App.get_center_of_points(heel, footindex)
        circle_radius = App.get_distance(heel, footindex) / 2
        center_pos = (math.floor(center_pos[0] * width), math.floor(center_pos[1] * height))
        circle_radius = math.floor(circle_radius * width)
        return (center_pos, circle_radius)


    def inpaint_foots(self, source:np.ndarray, segmentation_mask:np.ndarray, landmarks:List[Any]) -> np.ndarray:
        """
        Apply inpainting to foots.

        Args:
          source: Target image (BGR color as uint8)
          segmentation_mask: mask of human (Grayscale as uint8)
          landmarks: landmark list (results.pose_landmarks.landmark)

        Returns:
          Processed image as np.ndarray
        """

        height, width, _channel = source.shape

        left_center_pos, left_circle_radius = self.get_center_and_radious_for_foot(width, height, landmarks,
                                                                                   mp_pose.PoseLandmark.LEFT_ANKLE,mp_pose.PoseLandmark.LEFT_HEEL,mp_pose.PoseLandmark.LEFT_FOOT_INDEX)

        right_center_pos, right_circle_radius = self.get_center_and_radious_for_foot(width, height, landmarks,
                                                                                     mp_pose.PoseLandmark.RIGHT_ANKLE,mp_pose.PoseLandmark.RIGHT_HEEL,mp_pose.PoseLandmark.RIGHT_FOOT_INDEX)


        foots_circled_mask = np.zeros((height, width), dtype=np.uint8)

        cv2.circle(foots_circled_mask, center=left_center_pos, radius=left_circle_radius, color=255, thickness=-1)
        cv2.circle(foots_circled_mask, center=right_center_pos, radius=right_circle_radius, color=255, thickness=-1)

        mask = cv2.bitwise_and(segmentation_mask, foots_circled_mask)

        inpaintradius = 4

        left_roi_top = math.floor(max(0, left_center_pos[1] - left_circle_radius * 2.5))
        left_roi_left = math.floor(max(0, left_center_pos[0] - left_circle_radius * 2.5))
        left_roi_right = math.floor(min(width, left_center_pos[0] + left_circle_radius * 2.5))
        left_roi_bottom = math.floor(min(height, left_center_pos[1] + left_circle_radius * 2.5))

        left_roi_image = source[left_roi_top:left_roi_bottom, left_roi_left:left_roi_right]
        left_roi_mask = mask[left_roi_top:left_roi_bottom, left_roi_left:left_roi_right]

        if np.any(left_roi_mask):
            left_inpainted = cv2.inpaint(left_roi_image, left_roi_mask, inpaintradius, cv2.INPAINT_TELEA)
            source[left_roi_top:left_roi_bottom, left_roi_left:left_roi_right] = left_inpainted

        right_roi_top = math.floor(max(0, right_center_pos[1] - right_circle_radius * 2.5))
        right_roi_left = math.floor(max(0, right_center_pos[0] - right_circle_radius * 2.5))
        right_roi_right = math.floor(min(width, right_center_pos[0] + right_circle_radius * 2.5))
        right_roi_bottom = math.floor(min(height, right_center_pos[1] + right_circle_radius * 2.5))

        right_roi_image = source[right_roi_top:right_roi_bottom, right_roi_left:right_roi_right]
        right_roi_mask = mask[right_roi_top:right_roi_bottom, right_roi_left:right_roi_right]

        if np.any(right_roi_mask):
            right_inpainted = cv2.inpaint(right_roi_image, right_roi_mask, inpaintradius, cv2.INPAINT_TELEA)
            source[right_roi_top:right_roi_bottom, right_roi_left:right_roi_right] = right_inpainted

        return source


    def postprocessing_thread_main(self, videoframesize:Tuple[int, int], inqueue:Queue, outqueue:Queue) -> None:

        if self.options.mask_expansion >= 1:
            kernel_size = math.floor(self.options.mask_expansion) + 1
            dilate_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        else:
            dilate_kernel = None                
        maskthreshold = max(min(self.options.mask_threshold, 1), 0)

        landmarkstyle = mp_drawing_styles.get_default_pose_landmarks_style()
        for v in landmarkstyle.values():
            v.circle_radius = math.floor(videoframesize[0] * self.options.bone_width / 2)
        # NOTE: Specify the color in BGR
        colortable = { "white": (255,255,255), "black":"0,0,0", "red":(0,0,255), "green":(0,255,0), "blue":(255,0,0)}
        connectionscolor = colortable[self.options.bone_color]
        connectionsstyle = mp_drawing_styles.DrawingSpec(color=connectionscolor, thickness=math.floor(videoframesize[0] * self.options.bone_width))

        try:
            while True:
                newitem:Optional[Tuple[np.ndarray, Any]] = inqueue.get()
                if newitem is None:
                    break
                
                # Execute post processing
                frame, results = newitem

                if results is not None:
                    # Segmentation mask is 0-1.0
                    mask = results.segmentation_mask
                    # mask = np.where(mask <= maskthreshold, 255, 0).astype(np.uint8)
                    mask = np.where(mask <= maskthreshold, 0, 255).astype(np.uint8)

                    if dilate_kernel is not None:
                        # Expand area
                        mask = cv2.dilate(mask, dilate_kernel, iterations=1)

                    # mask2 = np.where(mask != 0, 0, 255).astype(np.uint8)
                    mask2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                    # Update frame
                    frame = np.where(mask2 != 0, self.prev_frame, frame)

                    if self.options.use_inpaint_foots:
                        self.inpaint_foots(frame, mask, results.pose_landmarks.landmark)

                    # Copy as previous frame for next loop
                    self.prev_frame[:,:,:] = frame

                    # Draw bones on frame if requested.
                    if self.options.draw_bones:
                        mp_drawing.draw_landmarks(
                            frame,
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                            # landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                            landmark_drawing_spec=landmarkstyle,
                            connection_drawing_spec=connectionsstyle
                        )

                # Pass to encoding thread
                outqueue.put(frame)

        except:
            traceback.print_exc()
            return


    def encoding_thread_main(self, queue:Queue, videowriter:cv2.VideoWriter) -> None:
        try:
            while True:
                newitem:Optional[np.ndarray] = queue.get()
                if newitem is None:
                    # None is a mark for exit.
                    break

                # newitem is a new frame to be encoded
                videowriter.write(newitem)

            return

        except:
            traceback.print_exc()
            return


    def main(self) -> NoReturn:
        """
        App's main routine.
        """

        pprint(self.options)
        if not self.options.input or not os.path.isfile(self.options.input):
            print(f'Input file not found: "{self.options.input}"', file=sys.stderr)
            exit(-1)

        if self.options.input == self.options.output:
            print(f'Input and output is identical. Stop.')
            exit(-2)

        if not self.options.allow_overwrite and os.path.isfile(self.options.output):
            print(f'Output file "{self.options.output}" already exists. Stop.')
            print(f'NOTE: Use "--allow-overwrite" ("-y") option if you intend.')
            exit(-3)

        cap = cv2.VideoCapture(self.options.input)
        videofps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = math.floor(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        videoframesize = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        videoframesize = (math.floor(videoframesize[0]), math.floor(videoframesize[1]))
        videofourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(self.options.output, videofourcc, videofps, videoframesize, True)

        exit_code = -8

        self.queue1 = Queue(64)
        self.queue2 = Queue(64)

        self.postprocess_thread = threading.Thread(target=self.postprocessing_thread_main, args=(videoframesize, self.queue1, self.queue2))
        self.postprocess_thread.start()

        self.encode_thread = threading.Thread(target=self.encoding_thread_main, args=(self.queue2, writer))
        self.encode_thread.start()

        print('Processing...')

        try:
            self.prev_frame = self.draw_graysquareboxpattern(videoframesize[0], videoframesize[1])
            prev_status_update_time = time.time()
            start_time = time.time()
            processed_frames = 0
            # Process 8 frames at a time with MediaPipe and pass to postprocessing ; value based on rough measurements.
            frame_block_size = 8
            frames_buffer = []
            is_last_frame = False


            # Init MediaPipe Pose estimation engine
            with mp_pose.Pose(
                model_complexity=int(self.options.model_complexity),
                enable_segmentation=True,
                min_detection_confidence=self.options.detection_threshold,
                min_tracking_confidence=self.options.tracking_threshold
            ) as pose:

                # Continue while source video is available.
                while True:
                    (success, frame) = cap.read()
                    if not success:
                        is_last_frame = True

                    else:
                        # To improve performance, optionally mark the image as not writeable to pass by reference.
                        frame.flags.writeable = False
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Execute MediaPipe's estimation
                        results = pose.process(frame_rgb)

                        processed_frames += 1

                        if not results or results.segmentation_mask is None or len(results.segmentation_mask) < 1:
                            # If estimation result is empty, write frame as original.
                            frames_buffer.append((frame, None))
                        
                        else:
                            # On estimation succeeded
                            frames_buffer.append((frame, results))


                    if is_last_frame or len(frames_buffer) == frame_block_size:
                        try:
                            while True:
                                f = frames_buffer.pop(0)
                                # writer.write(f)
                                self.queue1.put(f)
                        except IndexError:
                            pass

                    # Print statistics in interval
                    if is_last_frame or (time.time() - prev_status_update_time > 1):
                        sys.stdout.buffer.write(b'\r')
                        sys.stdout.flush()
                        elapsed_time_sec = time.time() - start_time
                        fps = processed_frames / elapsed_time_sec
                        msg = f'FPS: {fps:0.2f}, Frames: {processed_frames}/{total_frames}, Time: {elapsed_time_sec:0.2f}[sec]'
                        msg += ' ' * 4
                        print(msg, end='')
                        prev_status_update_time = time.time()
                        sys.stdout.flush()

                    if is_last_frame:
                        exit_code = 0
                        raise App.Finished

        except KeyboardInterrupt:
            print("")
            # Write the frames in buffer
            try:
                while True:
                    f = frames_buffer.pop(0)
                    self.queue1.put(f)
            except IndexError:
                pass
            print('Interrupted by user.')
            exit_code = 0

        except App.Finished:
            print("")
            exit_code = 0

        except Exception as excep:
            print("")
            traceback.print_exc()
            print('Exit by unexpected error.')
            exit_code = -16

        finally:
            self.queue1.put(None)
            self.postprocess_thread.join()

            self.queue2.put(None)
            self.encode_thread.join()

            cap.release()
            writer.release()

        exit(exit_code)


app = App()
app.main()
