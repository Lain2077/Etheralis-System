#This is a modified version of the FastMTCNN algorithm.
#Modifications include:
##1. Added support for CUDA
##2. Added support for boxes coordinates return

import cv2
from facenet_pytorch import MTCNN
import torch

class FastMTCNN(object):
    """Fast MTCNN implementation."""

    def __init__(self, stride, resize=1, *args, **kwargs):
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)

    def __call__(self, frames):
        cv2_cuda_enabled = cv2.cuda.getCudaEnabledDeviceCount() > 0
        """Detect faces in frames using strided MTCNN."""
        processed_frames = frames
        if self.resize != 1 and cv2_cuda_enabled:
            processed_frames = []
            for f in frames:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(f)
                if self.resize != 1:
                    dsize = (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize))
                    gpu_frame = cv2.cuda.resize(gpu_frame, dsize)
                processed_frames.append(gpu_frame.download())
        else:
            processed_frames = [cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize))) for f in frames]

        boxes, _ = self.mtcnn.detect(processed_frames[::self.stride])
        faces = []
        if boxes is not None:
            for i, (frame, frame_boxes) in enumerate(zip(processed_frames, boxes)):
                if frame_boxes is not None:
                    for box in frame_boxes:
                        x1, y1, x2, y2 = [int(b) for b in box]
                        faces.append(frame[y1:y2, x1:x2])
        return faces, boxes