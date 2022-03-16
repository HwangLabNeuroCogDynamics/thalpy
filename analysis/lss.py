from thalpy import masks
from thalpy.constants.paths import SCRIPTS_DIR
import nibabel as nib
import numpy as np
import os
import sys
import pandas as pd
import pickle


class SubjectLssTent:
    def __init__(self, sub_deconvolve_dir, cues, tent_length=9, path="LSS_TENT.p"):
        self.cues = cues
        self.tent_length = tent_length
        self.sub_deconvolve_dir = sub_deconvolve_dir

        lss_img = nib.load(
            os.path.join(sub_deconvolve_dir, self.cues[0] + ".LSS+tlrc.BRIK")
        )
        lss_sample_data = lss_img.get_fdata()
        self.affine = lss_img.affine

        self.trial_df = pd.read_csv(
            os.path.join(self.sub_deconvolve_dir, "conditions.csv")
        )
        self.num_trials = len(self.trial_df.index)
        self.path = os.path.join(self.sub_deconvolve_dir, path)
        self.voxel_shape = lss_sample_data.shape[:3]

        self.__convert_lss_files()
        self.__avg_tent_matrix()
        self.save()

    @staticmethod
    def load(filepath):
        sys.path.append(os.path.join(SCRIPTS_DIR, "thalhi/"))
        return pickle.load(open(filepath, "rb"))

    def save(self, path=None):
        if path:
            self.path = path
        pickle.dump(self, open(self.path, "wb"), protocol=4)

    def __convert_lss_files(self):
        self.data = np.ones(
            [
                self.voxel_shape[0],
                self.voxel_shape[1],
                self.voxel_shape[2],
                self.tent_length * self.num_trials,
            ]
        )

        for cue, group in self.trial_df.groupby(["Cue"]):
            lss_data = nib.load(
                self.sub_deconvolve_dir + f"{cue}.LSS+tlrc.BRIK"
            ).get_fdata()

            if lss_data.shape[:3] != self.data.shape[:3]:
                raise Exception(
                    "Voxel shape between lss_data and trial matrix does not match."
                )

            for brik_idx, (trial_idx, trial) in enumerate(group.iterrows()):
                trial_tent_idx = trial_idx * self.tent_length
                brik_tent_idx = brik_idx * self.tent_length
                self.data[
                    :, :, :, trial_tent_idx: trial_tent_idx + self.tent_length
                ] = lss_data[:, :, :, brik_tent_idx: brik_tent_idx + self.tent_length]

    def __avg_tent_matrix(self):
        self.avg_data = np.ones(
            [
                self.voxel_shape[0],
                self.voxel_shape[1],
                self.voxel_shape[2],
                self.num_trials,
            ]
        )
        for trial in range(self.num_trials):
            self.avg_data[:, :, :, trial] = np.nanmean(
                self.data[:, :, :, trial * 9: (trial + 1) * 9], axis=-1
            )

    def mask_rois(self, mask_path):
        nii_img = nib.Nifti1Image(self.data, self.affine)
        self.rois = []
        num_rois = masks.masker_count(masks.binary_masker(mask_path))
        for i in range(num_rois):
            masker = masks.binary_masker(
                mask_path, img_math=f"img=={i + 1}"
            )
            self.rois.append(masker.fit_transform(nii_img))

    def remove_nan_trials(self):
        censor_array = np.ones([self.num_trials * self.tent_length])
        df_censor_array = np.ones([self.num_trials])

        for trial in range(self.num_trials):
            trial_tent_index = trial * self.tent_length
            if np.any(np.isnan(self.data[:, :, :, trial_tent_index: trial_tent_index + self.tent_length])):
                censor_array[trial_tent_index: trial_tent_index + self.tent_length] = 0
                df_censor_array[trial] = 0
                
        self.data = np.delete(self.data, censor_array, axis=-1)
        self.trial_df = self.trial_df.drop(df_censor_array)
        self.num_trials = len(self.trial_df.index)
        
