import cv2
import numpy as np
import os 

class DataGenerator:
    def __init__(self, shape, dataset_path, length = 5, stride = 6):
        self.stable_path = os.path.join(dataset_path,'stable')
        self.unstable_path = os.path.join(dataset_path,'unstable')
        self.flows_path = os.path.join(dataset_path,'Flows')
        self.kpts_path = os.path.join(dataset_path,'matched_kpts')
        self.shape = shape
        self.length = length
        self.stride = stride
        self.shape = shape
        self.video_names = os.listdir(self.stable_path)
        self.frame_idx = 30

    def get_paths(self,video):
        s_path = os.path.join(self.stable_path,video)
        u_path = os.path.join(self.unstable_path,video)
        flow_path = os.path.join(self.flows_path,video[:-4]+'.npy')
        kpt_path = os.path.join(self.kpts_path,video[:-4]+'.npy')
        paths = [s_path,u_path,flow_path,kpt_path]
        return( paths)

    def __call__(self):
        for video in self.video_names:
            paths = self.get_paths(video)
            stable_frames, unstable_frames, flows, kpts = load_video(paths,self.shape)
            n,h,w,c = stable_frames.shape
            sequence_curr = np.zeros(shape=(h,w,self.length),dtype=np.float32)
            sequence_prev = np.zeros_like(sequence_curr)
            for frame_idx in range(30,n):
                for (i,j) in zip(range(frame_idx - self.stride, frame_idx - self.length*self.stride -1, -self.stride) , range(self.length)):
                    sequence_curr[:,:,j] = cv2.cvtColor(stable_frames[i,...],cv2.COLOR_BGR2GRAY)
                    sequence_prev[:,:,j] = cv2.cvtColor(stable_frames[i - 1,...],cv2.COLOR_BGR2GRAY)
                It_curr = unstable_frames[frame_idx,...]
                Igt_curr = stable_frames[frame_idx,...]
                It_prev = unstable_frames[frame_idx - 1,...]
                flow = flows[frame_idx]
                kpt = kpts[frame_idx]
                yield sequence_curr, sequence_prev, Igt_curr,It_curr ,It_prev, flow , kpt

def load_video(paths,shape):
    stable_frames = []
    unstable_frames = []
    stable_cap = cv2.VideoCapture(paths[0])
    unstable_cap = cv2.VideoCapture(paths[1])
    flows = np.load(paths[2])
    kpts = np.load(paths[3])
    while True:
        ret, frame1 = stable_cap.read()
        if not ret:
            break
        frame1 = preprocess(frame1,shape)
        stable_frames.append(frame1)
        ret, frame2 = unstable_cap.read()
        if not ret:
            break
        frame2 = preprocess(frame2,shape)
        unstable_frames.append(frame2)
    stable_cap.release()
    unstable_cap.release()
    #in some video pairs the stable and unstable version dont have the same frame count
    frame_count = min(len(stable_frames),len(unstable_frames))
    stable_frames = stable_frames[:frame_count]
    unstable_frames = unstable_frames[:frame_count]
    #convert to np.arrays
    stable_frames = np.array(stable_frames,dtype=np.float32)
    unstable_frames = np.array(unstable_frames,dtype=np.float32)
    return(stable_frames,unstable_frames,flows,kpts)

def preprocess(img,shape):
    h,w,_ = shape
    img = cv2.resize(img,(w,h),cv2.INTER_LINEAR)
    img = (img- 127.0) / 127.0
    return img