import copy
import time
import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend import config
from backend.inpaint.sttn.auto_sttn import InpaintGenerator
from backend.inpaint.utils.sttn_utils import Stack, ToTorchFormatTensor

_to_tensors = transforms.Compose([Stack(), ToTorchFormatTensor()])


class STTNInpaint:
    def __init__(self):
        self.device = config.device
        self.model = InpaintGenerator().to(self.device)
        self.model.load_state_dict(torch.load(config.STTN_MODEL_PATH, map_location=self.device)['netG'])
        self.model.eval()
        self.model_input_width, self.model_input_height = 640, 120
        self.neighbor_stride = config.STTN_NEIGHBOR_STRIDE
        self.ref_length = config.STTN_REFERENCE_LENGTH

    def __call__(self, input_frames: List[np.ndarray], input_mask: np.ndarray):
        _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
        mask = mask[:, :, None]
        H_ori, W_ori = mask.shape[:2]
        H_ori = int(H_ori + 0.5)
        W_ori = int(W_ori + 0.5)
        split_h = int(W_ori * 3 / 16)
        inpaint_area = self.get_inpaint_area_by_mask(H_ori, split_h, mask)

        if not inpaint_area:
            print("[Warning] No inpaint area detected. Processing full frame.")
            inpaint_area = [(0, H_ori)]

        frames_hr = copy.deepcopy(input_frames)
        frames_scaled = {}
        comps = {}
        inpainted_frames = []

        for k in range(len(inpaint_area)):
            frames_scaled[k] = []

        for j in range(len(frames_hr)):
            image = frames_hr[j]
            for k in range(len(inpaint_area)):
                image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                image_resize = cv2.resize(image_crop, (self.model_input_width, self.model_input_height))
                frames_scaled[k].append(image_resize)

        for k in range(len(inpaint_area)):
            comps[k] = self.inpaint(frames_scaled[k])

        if inpaint_area:
            for j in range(len(frames_hr)):
                frame = frames_hr[j]
                for k in range(len(inpaint_area)):
                    comp = cv2.resize(comps[k][j], (W_ori, split_h if inpaint_area[k][1] - inpaint_area[k][0] == split_h else inpaint_area[k][1] - inpaint_area[k][0]))
                    comp = cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB)
                    mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], :]
                    frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = mask_area * comp + (1 - mask_area) * frame[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                inpainted_frames.append(frame)
                print(f'processing frame, {len(frames_hr) - j} left')
        return inpainted_frames

    @staticmethod
    def read_mask(path):
        img = cv2.imread(path, 0)
        ret, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        img = img[:, :, None]
        return img

    def get_ref_index(self, neighbor_ids, length):
        ref_index = []
        for i in range(0, length, self.ref_length):
            if i not in neighbor_ids:
                ref_index.append(i)
        return ref_index

    def inpaint(self, frames: List[np.ndarray]):
        frame_length = len(frames)
        feats = _to_tensors(frames).unsqueeze(0) * 2 - 1
        feats = feats.to(self.device)
        comp_frames = [None] * frame_length
        with torch.no_grad():
            feats = self.model.encoder(feats.view(frame_length, 3, self.model_input_height, self.model_input_width))
            _, c, feat_h, feat_w = feats.size()
            feats = feats.view(1, frame_length, c, feat_h, feat_w)
        for f in range(0, frame_length, self.neighbor_stride):
            neighbor_ids = [i for i in range(max(0, f - self.neighbor_stride), min(frame_length, f + self.neighbor_stride + 1))]
            ref_ids = self.get_ref_index(neighbor_ids, frame_length)
            with torch.no_grad():
                pred_feat = self.model.infer(feats[0, neighbor_ids + ref_ids, :, :, :])
                pred_img = torch.tanh(self.model.decoder(pred_feat[:len(neighbor_ids), :, :, :])).detach()
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8)
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
        return comp_frames

    @staticmethod
    def get_inpaint_area_by_mask(H, h, mask):
        inpaint_area = []
        to_H = from_H = H
        while from_H != 0:
            if to_H - h < 0:
                from_H = 0
                to_H = h
            else:
                from_H = to_H - h
            if not np.all(mask[from_H:to_H, :] == 0) and np.sum(mask[from_H:to_H, :]) > 10:
                if to_H != H:
                    move = 0
                    while to_H + move < H and not np.all(mask[to_H + move, :] == 0):
                        move += 1
                    if to_H + move < H and move < h:
                        to_H += move
                        from_H += move
                if (from_H, to_H) not in inpaint_area:
                    inpaint_area.append((from_H, to_H))
                else:
                    break
            to_H -= h
        return inpaint_area

    @staticmethod
    def get_inpaint_area_by_selection(input_sub_area, mask):
        print('use selection area for inpainting')
        height, width = mask.shape[:2]
        ymin, ymax, _, _ = input_sub_area
        interval_size = 135
        inpaint_area = []
        for i in range(ymin, ymax, interval_size):
            inpaint_area.append((i, i + interval_size))
        if inpaint_area[-1][1] != ymax:
            if inpaint_area[-1][1] + interval_size <= height:
                inpaint_area.append((inpaint_area[-1][1], inpaint_area[-1][1] + interval_size))
        return inpaint_area


class STTNVideoInpaint:
    def read_frame_info_from_video(self):
        reader = cv2.VideoCapture(self.video_path)
        frame_info = {
            'W_ori': int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5),
            'H_ori': int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5),
            'fps': reader.get(cv2.CAP_PROP_FPS),
            'len': int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        }
        return reader, frame_info

    def __init__(self, video_path, mask_path=None, clip_gap=None):
        self.sttn_inpaint = STTNInpaint()
        self.video_path = video_path
        self.mask_path = mask_path
        self.video_out_path = os.path.join(
            os.path.dirname(os.path.abspath(self.video_path)),
            f"{os.path.basename(self.video_path).rsplit('.', 1)[0]}_no_sub.mp4"
        )
        if clip_gap is None:
            self.clip_gap = config.STTN_MAX_LOAD_NUM
        else:
            self.clip_gap = clip_gap

    def __call__(self, input_mask=None, input_sub_remover=None, tbar=None):
        reader, frame_info = self.read_frame_info_from_video()
        if input_sub_remover is not None:
            writer = input_sub_remover.video_writer
        else:
            writer = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_info['fps'], (frame_info['W_ori'], frame_info['H_ori']))

        rec_time = frame_info['len'] // self.clip_gap if frame_info['len'] % self.clip_gap == 0 else frame_info['len'] // self.clip_gap + 1
        split_h = int(frame_info['W_ori'] * 3 / 16)

        if input_mask is None and self.mask_path:
            mask = self.sttn_inpaint.read_mask(self.mask_path)
        elif input_mask is None:
            print("[Info] No mask provided. Creating full-screen mask.")
            mask = np.ones((frame_info['H_ori'], frame_info['W_ori']), dtype=np.uint8) * 255
            _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
            mask = mask[:, :, None]
        else:
            _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
            mask = mask[:, :, None]

        inpaint_area = self.sttn_inpaint.get_inpaint_area_by_mask(frame_info['H_ori'], split_h, mask)
        if not inpaint_area:
            print("[Warning] No inpaint area detected. Processing full frame.")
            inpaint_area = [(0, frame_info['H_ori'])]

        for i in range(rec_time):
            start_f = i * self.clip_gap
            end_f = min((i + 1) * self.clip_gap, frame_info['len'])
            print('Processing:', start_f + 1, '-', end_f, ' / Total:', frame_info['len'])
            frames_hr = []
            frames = {}
            comps = {}

            # 读取帧并检查成功
            for j in range(start_f, end_f):
                success, image = reader.read()
                if not success:
                    print(f"[Warning] Failed to read frame {j + 1}. Stopping at {len(frames_hr)} frames.")
                    break
                frames_hr.append(image)

            # 如果没有读取到任何帧，跳过此次循环
            if not frames_hr:
                print("[Warning] No frames read in this segment. Skipping.")
                continue

            # 初始化帧字典
            for k in range(len(inpaint_area)):
                frames[k] = []

            # 填充 frames 字典
            for j in range(len(frames_hr)):
                image = frames_hr[j]
                for k in range(len(inpaint_area)):
                    image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                    image_resize = cv2.resize(image_crop, (self.sttn_inpaint.model_input_width, self.sttn_inpaint.model_input_height))
                    frames[k].append(image_resize)

            # 处理每个修复区域
            for k in range(len(inpaint_area)):
                comps[k] = self.sttn_inpaint.inpaint(frames[k])

            # 处理实际读取的帧
            for j in range(len(frames_hr)):
                if input_sub_remover is not None and input_sub_remover.gui_mode:
                    original_frame = copy.deepcopy(frames_hr[j])
                else:
                    original_frame = None
                frame = frames_hr[j]
                for k in range(len(inpaint_area)):
                    comp = cv2.resize(comps[k][j], (frame_info['W_ori'], split_h if inpaint_area[k][1] - inpaint_area[k][0] == split_h else inpaint_area[k][1] - inpaint_area[k][0]))
                    comp = cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB)
                    mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], :]
                    frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = mask_area * comp + (1 - mask_area) * frame[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                writer.write(frame)
                if input_sub_remover is not None:
                    if tbar is not None:
                        input_sub_remover.update_progress(tbar, increment=1)
                    if original_frame is not None and input_sub_remover.gui_mode:
                        input_sub_remover.preview_frame = cv2.hconcat([original_frame, frame])

        writer.release()
        reader.release()


if __name__ == '__main__':
    mask_path = '../../test/test.png'
    video_path = '../../test/test.mp4'
    start = time.time()
    sttn_video_inpaint = STTNVideoInpaint(video_path, mask_path, clip_gap=config.STTN_MAX_LOAD_NUM)
    sttn_video_inpaint()
    print(f'video generated at {sttn_video_inpaint.video_out_path}')
    print(f'time cost: {time.time() - start}')
