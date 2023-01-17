import os
import pandas as pd
import pickle
import librosa
import numpy as np
import torch
import torch.utils.data

import slowfast.utils.logging as logging

from .build import DATASET_REGISTRY

from .spec_augment import combined_transforms
from . import utils as utils
from .audio_loader_vggsound import get_start_end_idx_3d, _extract_sound_feature_3d, _log_specgram

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Vggsound3d(torch.utils.data.Dataset):
    def __init__(self, cfg, mode):
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for VGG-Sound".format(mode)
        self.cfg = cfg
        self.mode = mode
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS

        logger.info("Constructing VGG-Sound {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the audio loader.
        """
        if self.mode == "train":
            path_annotations_pickle = os.path.join(self.cfg.VGGSOUND.ANNOTATIONS_DIR, self.cfg.VGGSOUND.TRAIN_LIST)
        elif self.mode == "val":
            path_annotations_pickle = os.path.join(self.cfg.VGGSOUND.ANNOTATIONS_DIR, self.cfg.VGGSOUND.VAL_LIST)
        else:
            path_annotations_pickle = os.path.join(self.cfg.VGGSOUND.ANNOTATIONS_DIR, self.cfg.VGGSOUND.TEST_LIST)

        assert os.path.exists(path_annotations_pickle), "{} dir not found".format(path_annotations_pickle)

        self._audio_records = []
        self._temporal_idx = []
        for tup in pd.read_pickle(path_annotations_pickle).iterrows():
            for idx in range(self._num_clips):
                self._audio_records.append(tup[1])
                self._temporal_idx.append(idx)
        assert (len(self._audio_records) > 0), "Failed to load VGG-Sound split {} from {}".format(self.mode, path_annotations_pickle)
        logger.info("Constructing vggsound dataloader (size: {}) from {}".format(len(self._audio_records), path_annotations_pickle))

    def __getitem__(self, index):
        """
        Given the audio index, return the spectrogram, label, and audio
        index.
        Args:
            index (int): the audio index provided by the pytorch sampler.
        Returns:
            spectrogram (tensor): the spectrogram sampled from the audio. The dimension
                is `channel` x `num frames` x `num frequencies`.
            label (int): the label of the current audio.
            index (int): Return the index of the audio.
        """
        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
        elif self.mode in ["test"]:
            temporal_sample_index = self._temporal_idx[index]
        else:
            raise NotImplementedError("Does not support {} mode".format(self.mode))

        path_audio = os.path.join(self.cfg.VGGSOUND.AUDIO_DATA_DIR, self._audio_records[index]['video'][:-4] + '.wav')
        samples, sr = librosa.core.load(path_audio, sr=None, mono=False)
        assert sr == self.cfg.AUDIO_DATA.SAMPLING_RATE, "Audio sampling rate ({}) does not match target sampling rate ({})".format(sr, self.cfg.AUDIO_DATA.SAMPLING_RATE)
        start_indices, end_indices = get_start_end_idx_3d(
            audio_size=samples.shape[0],
            frame_size=int(round(self.cfg.AUDIO_DATA.SAMPLING_RATE * self.cfg.AUDIO_DATA.CLIP_SECS)),
            frame_interval=self.cfg.AUDIO_DATA.INTERVAL,
            num_frames=self.cfg.AUDIO_DATA.NUM_FRAME_CROPS,
            clip_idx=temporal_sample_index,
            num_clips=self.cfg.TEST.NUM_ENSEMBLE_VIEWS
        )

        spectrogram = list()
        for start_idx, end_idx in zip(start_indices, end_indices):
            if samples.shape[0] <= int(start_idx):
                spec = spectrogram[-1]
            else:
                spec = _extract_sound_feature_3d(self.cfg, samples, int(start_idx), int(end_idx))
                spec = spec.float()
                # C T F -> C F T
                spec = spec.permute(0, 2, 1)
                if self.mode in ["train"]:
                    # SpecAugment
                    spec = combined_transforms(spec)
            spectrogram.append(spec)
        # C F T -> C N F T
        spectrogram = torch.stack(spectrogram, dim=1)
        label = self._audio_records[index]['class_id']

        return spectrogram, label, index, {}

    def __len__(self):
        return len(self._audio_records)
