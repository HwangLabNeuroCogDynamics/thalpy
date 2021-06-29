import glob2 as glob
from thalpy.constants import paths
import os
import getpass
import pandas as pd


class Subjects(list):
    def to_subargs(self):
        return " ".join(sub.name for sub in self)

    def to_subargs_list(self):
        return [sub.name for sub in self]

    def get_sub_index(self, sub_name):
        for index, subject in enumerate(self):
            if sub_name == subject.name:
                return index
        return -1


class Subject:
    def get_sub_dir(self, path):
        return os.path.join(path, self.sub_dir)

    def get_sub_sessions(self, run_dir):
        return sorted(
            [dir for dir in os.listdir(run_dir + self.sub_dir) if "ses" in dir]
        )

    def __init__(self, name, dir_tree, run_dir, sessions=None):
        self.name = name

        # set directories
        self.sub_dir = f"{paths.SUB_PREFIX}{name}/"
        self.dataset_dir = dir_tree.dataset_dir
        self.bids_dir = self.get_sub_dir(dir_tree.bids_dir)
        self.mriqc_dir = self.get_sub_dir(dir_tree.mriqc_dir)
        self.fmriprep_dir = self.get_sub_dir(dir_tree.fmriprep_dir)
        self.deconvolve_dir = self.get_sub_dir(dir_tree.deconvolve_dir)
        self.fc_dir = self.get_sub_dir(dir_tree.fc_dir)

        # set sessions and runs
        if dir_tree.sessions:
            sub_sessions = self.get_sub_sessions(run_dir)
            self.sessions = [ses for ses in dir_tree.sessions if ses in sub_sessions]
        elif sessions is None:
            self.sessions = self.get_sub_sessions(run_dir)
        else:
            self.sessions = sessions
        self.runs = []


# Functions
def get_sub_dir(subject):
    return f"{paths.SUB_PREFIX}{subject}/"


def get_subjects(subject_dir, dir_tree, sessions=None, completed_subs=None, num=None):
    """[summary]

    Args:
        subject_dir ([type]): [description]
        dir_tree ([type]): [description]
        sessions ([type], optional): [description]. Defaults to None.
        completed_subs ([type], optional): [description]. Defaults to None.
        num ([type], optional): [description]. Defaults to None.

    Returns:
        [Subject] list of subjects
    """
    subargs = get_subargs(subject_dir, completed_subs=completed_subs, num=num)
    return subargs_to_subjects(subargs, dir_tree, subject_dir, sessions=sessions)


def read_file_subargs(filepath, dir_tree, num=None):
    with open(filepath) as file:
        subargs = file.read().splitlines()
    if num:
        subargs = subargs[:num]
    return subargs_to_subjects(subargs, dir_tree, dir_tree.fmriprep_dir)


def get_subargs(sub_dir, completed_subs=None, num=None):
    subargs = [
        dir.replace(paths.SUB_PREFIX, "")
        for dir in os.listdir(sub_dir)
        if os.path.isdir(os.path.join(sub_dir, dir)) and paths.SUB_PREFIX in dir
    ]
    if completed_subs:
        subargs = sorted(
            [sub for sub in subargs if sub not in completed_subs.to_subargs_list()]
        )
    subargs = sorted(subargs)
    if num:
        return subargs[:num]
    else:
        return subargs


def subargs_to_subjects(subargs, dir_tree, sub_dir, sessions=None):
    subjects = Subjects()
    for sub in subargs:
        subjects.append(Subject(sub, dir_tree, sub_dir, sessions=sessions))
    return subjects


def get_sub_runs(sub_bids_dir):
    runs = []

    return runs


class DirectoryTree:
    def __init__(self, dataset_dir, bids_dir=None, work_dir=None, sessions=None):
        dataset_dir = check_trailing_slash(dataset_dir)
        bids_dir = check_trailing_slash(bids_dir)
        work_dir = check_trailing_slash(work_dir, exists=False)

        self.dataset_name = os.path.basename(os.path.normpath(dataset_dir))
        self.dataset_dir = dataset_dir
        self.mriqc_dir = dataset_dir + paths.MRIQC_DIR
        self.fmriprep_dir = dataset_dir + paths.FMRIPREP_DIR
        self.deconvolve_dir = dataset_dir + paths.DECONVOLVE_DIR
        self.raw_dir = dataset_dir + paths.RAW_DIR
        self.analysis_dir = dataset_dir + paths.ANALYSIS_DIR
        self.fc_dir = dataset_dir + paths.FC_DIR
        self.log_dir = dataset_dir + paths.LOGS_DIR
        self.sessions = sessions
        if bids_dir is None:
            self.bids_dir = dataset_dir + paths.BIDS_DIR
        else:
            self.bids_dir = bids_dir
        if work_dir is None:
            self.work_dir = f"{paths.LOCALSCRATCH}{getpass.getuser()}/"
        else:
            self.work_dir = work_dir


# Functions
def check_trailing_slash(filepath, exists=True):
    if filepath is None:
        return None
    elif filepath[-1] != "/":
        filepath += "/"
    if not os.path.exists(filepath) and exists is True:
        raise FileNotFoundError(filepath)
    return filepath


def get_ses_files(subject, run_file_dir, file_wc):
    """[summary]

    Args:
        subject ([type]): Subject with files
        run_file_dir ([type]): directory to search for files
        file_wc ([type]): wildcard to match file

    Returns:
        [Files]: List of files matching pattern
    """

    if subject.sub_dir.replace("/", "") in os.listdir(run_file_dir):
        run_file_dir = os.path.join(run_file_dir, subject.sub_dir)
    else:
        file_wc = f"*{subject.name}{file_wc}"

    print(run_file_dir)

    if not subject.sessions:
        if "func" in os.listdir(run_file_dir):
            file_pattern = os.path.join(run_file_dir, paths.FUNC_DIR, file_wc)
        else:
            file_pattern = os.path.join(run_file_dir, file_wc)
            print(file_pattern)
        files = sorted(glob.glob(file_pattern))
    else:
        for session in subject.sessions:
            if "func" in os.listdir(run_file_dir):
                file_pattern = os.path.join(
                    run_file_dir, session, paths.FUNC_DIR, file_wc
                )
            else:
                file_pattern = os.path.join(run_file_dir, session, file_wc)

            files.extend(sorted(glob.glob(file_pattern)))

    return files


def parse_sub_from_file(filepath):
    sub_string = ""
    split_string = filepath.split(paths.SUB_PREFIX)[1]
    for char in split_string:
        if not char.isdigit():
            break
        else:
            sub_string += char
    return sub_string


def parse_ses_from_file(filepath):
    ses_string = ""
    split_string = filepath.split("ses-")[1]
    for char in split_string:
        if char == "_" or char == "/":
            break
        else:
            ses_string += char
    return ses_string


def parse_run_from_file(filepath):
    run_string = ""
    split_string = filepath.split("run-")[1]
    for char in split_string:
        if not char.isdigit():
            break
        else:
            run_string += char
    return run_string


def parse_dir_from_file(filepath):
    dir_string = ""
    split_string = filepath.split("dir-")[1]
    for char in split_string:
        if char == "_":
            break
        else:
            dir_string += char
    return dir_string
