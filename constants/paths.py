import os

if os.path.exists("/mnt/nfs/lss/lss_kahwang_hpc/"):
    SCRIPTS_DIR = "/mnt/nfs/lss/lss_kahwang_hpc/scripts/"
elif os.path.exists("/Shared/lss_kahwang_hpc/"):
    SCRIPTS_DIR = "/Shared/lss_kahwang_hpc/scripts/"
else:
    Exception("Path not found.")

# Base paths
BIDS_DIR = "BIDS/"
SUB_PREFIX = "sub-"
FUNC_DIR = "func/"
MRIQC_DIR = "mriqc/"
FMRIPREP_DIR = "fmriprep/"
ANALYSIS_DIR = "analysis/"
DECONVOLVE_DIR = "3dDeconvolve/"
FREESURFER_DIR = "freesurfer/"
FC_DIR = "fc/"
LOGS_DIR = "logs/"
RAW_DIR = "Raw/"
WORK_DIR = "work/"
FMAP_DIR = "fmap/"
LOCALSCRATCH = "/localscratch/Users/"

# Base files
DATASET_DESCRIPTION = "dataset_description.json"
STIM_CONFIG = "stim_config.csv"
PARTICIPANTS_TSV = "participants.tsv"
REGRESSOR_FILE = "nuisance.1D"
CENSOR_FILE = "censor.1D"

# File suffixes
FC_SUFFIX = "_fc.nii"

# Keywords
SESSION = "SESSION"
