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

LICENSE_DESCRIPTION = "Licensed under MIT License (C) 2023 verylowfreq"


from typing import List, NoReturn, Optional
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

    MAXWIDTH_FOR_MP = 1024
    MAXHEIGHT_FOR_MP = 1024

    class Finished(Exception):
        pass

    def __init__(self) -> None:
        print(LICENSE_DESCRIPTION)
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
        self.argparser.add_argument('-b', '--draw-bones', action='store_true', help='For debug. Draw the detected bone lines')
        self.argparser.add_argument('-w', '--bone-width', type=float, default=App.DEFAULT_BONE_WIDTH, help=f'Width of bones in ratio to screen width (0.0-1.0; default {App.DEFAULT_BONE_WIDTH})')
        self.argparser.add_argument('-c', '--model-complexity', choices='012', default=App.DEFUALT_MODEL_COMPLEXITY, help='ML model complexity(0, 1, or 2 (more accurate, heavy))')
        self.argparser.add_argument('-y', '--allow-overwrite', action='store_true', help='Overwrite the output file if exists')
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

        self.queue = Queue(64)
        self.encode_thread = threading.Thread(target=self.encoding_thread_main, args=(self.queue, writer))
        self.encode_thread.start()

        print('Processing...')

        try:
            self.prev_frame = self.draw_graysquareboxpattern(videoframesize[0], videoframesize[1])
            prev_status_update_time = time.time()
            start_time = time.time()
            processed_frames = 0
            # frame_block_size = videofps
            frame_block_size = 8
            frames_buffer = []
            is_last_frame = False
            if self.options.mask_expansion >= 1:
                kernel_size = math.floor(self.options.mask_expansion) + 1
                dilate_kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            else:
                dilate_kernel = None                
            maskthreshold = max(min(self.options.mask_threshold, 1), 0)

            landmarkstyle = mp_drawing_styles.get_default_pose_landmarks_style()
            for v in landmarkstyle.values():
                v.circle_radius = math.floor(videoframesize[0] * self.options.bone_width / 2)
            connectionsstyle = mp_drawing_styles.DrawingSpec(thickness=math.floor(videoframesize[0] * self.options.bone_width))


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
                            pass
                        
                        else:
                            # On estimation succeeded

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

                        frames_buffer.append(frame)

                    if is_last_frame or len(frames_buffer) == frame_block_size:
                        try:
                            while True:
                                f = frames_buffer.pop(0)
                                # writer.write(f)
                                self.queue.put(f)
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
                    writer.write(f)
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
            self.queue.put(None)
            self.encode_thread.join()

            cap.release()
            writer.release()

        exit(exit_code)


app = App()
app.main()
