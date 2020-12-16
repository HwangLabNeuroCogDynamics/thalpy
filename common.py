import glob2 as glob
import basic_settings as s
import os
import getpass


# Subjects --------------------------------------------------------------------
# Classes
class Subjects(list):
    def to_subargs(self):
        return " ".join(sub.name for sub in self)

    def to_subargs_list(self):
        return [sub.name for sub in self]


class Subject:
    def __init__(self, name, dir_tree):
        self.name = name
        self.sub_dir = f'{s.SUB_PREFIX}{name}/'
        self.dataset_dir = dir_tree.dataset_dir
        self.bids_dir = dir_tree.bids_dir + self.sub_dir
        self.mriqc_dir = dir_tree.mriqc_dir + self.sub_dir
        self.fmriprep_dir = dir_tree.fmriprep_dir + self.sub_dir
        self.deconvolve_dir = dir_tree.deconvolve_dir + self.sub_dir
        # if dir_tree.sessions is None:
        #     self.sessions = get_sub_sessions(self.bids_dir)
        # else:
        self.sessions = dir_tree.sessions


# Functions
def get_sub_dir(subject):
    return f"{s.SUB_PREFIX}{subject}/"


def get_subjects(dir_tree, completed_subs=None, num=None):
    subargs = get_subargs(dir_tree.bids_dir, completed_subs=completed_subs,
                          num=num)
    return subargs_to_subjects(subargs, dir_tree)


def read_file_subargs(filepath, dir_tree, num=None):
    with open(filepath) as file:
        subargs = file.read().splitlines()
    if num:
        subargs = subargs[:num]
    return subargs_to_subjects(subargs, dir_tree)


def get_subargs(bids_dir, completed_subs=None, num=None):
    subargs = [dir.replace(s.SUB_PREFIX, '')
               for dir in os.listdir(bids_dir) if s.SUB_PREFIX in dir]
    if completed_subs:
        subargs = sorted([sub for sub in subargs
                          if sub not in completed_subs.to_subargs_list()])
    subargs = sorted(subargs)
    if num:
        return subargs[:num]
    else:
        return subargs


def subargs_to_subjects(subargs, dir_tree):
    subjects = Subjects()
    for sub in subargs:
        subjects.append(Subject(sub, dir_tree))
    return subjects


def get_sub_sessions(sub_bids_dir):
    return sorted([dir for dir in os.listdir(sub_bids_dir) if 'ses' in dir])


# Filepaths --------------------------------------------------------------------
# Classes
class DirectoryTree:
    def __init__(self, dataset_dir, bids_dir=None, work_dir=None, sessions=None):
        self.dataset_name = os.path.basename(os.path.normpath(dataset_dir))
        self.dataset_dir = dataset_dir
        self.mriqc_dir = dataset_dir + s.MRIQC_DIR
        self.fmriprep_dir = dataset_dir + s.FMRIPREP_DIR
        self.deconvolve_dir = dataset_dir + s.DECONVOLVE_DIR
        self.raw_dir = dataset_dir + s.RAW_DIR
        self.analysis_dir = dataset_dir + s.ANALYSIS_DIR
        self.log_dir = dataset_dir + s.LOGS_DIR
        self.sessions = sessions
        if bids_dir is None:
            self.bids_dir = dataset_dir + s.BIDS_DIR
        else:
            self.bids_dir = bids_dir
        if work_dir is None:
            self.work_dir = f'{s.LOCALSCRATCH}{getpass.getuser()}/'
        else:
            self.work_dir = work_dir


# Functions
def get_ses_files(sessions, pattern):
    if not sessions:
        new_pattern = pattern.replace(f'{s.SESSION}/', '')
        files = sorted(glob.glob(new_pattern))
        return files

    for session in sessions:
        new_pattern = pattern.replace(s.SESSION, session)
        return files.extend(sorted(glob.glob(new_pattern)))


def parse_sub_from_file(filepath):
    sub_string = ''
    split_string = filepath.split(s.SUB_PREFIX)[1]
    for char in split_string:
        if not char.isdigit():
            break
        else:
            sub_string += char
    return sub_string


def parse_ses_from_file(filepath):
    ses_string = ''
    split_string = filepath.split('ses-')[1]
    for char in split_string:
        if char == '_' or char == '/':
            break
        else:
            ses_string += char
    return ses_string


def parse_run_from_file(filepath):
    run_string = ''
    split_string = filepath.split('run-')[1]
    for char in split_string:
        if not char.isdigit():
            break
        else:
            run_string += char
    return run_string


def parse_dir_from_file(filepath):
    dir_string = ''
    split_string = filepath.split('dir-')[1]
    for char in split_string:
        if char == '_':
            break
        else:
            dir_string += char
    return dir_string


# Masks -----------------------------------------------------------------------
MOREL_DICT = {
    1: 'AN',
    2: 'VM',
    3: 'VL',
    4: 'MGN',
    5: 'MD',
    6: 'PuA',
    7: 'LP',
    8: 'IL',
    9: 'VA',
    10: 'Po',
    11: 'LGN',
    12: 'PuM',
    13: 'PuI',
    14: 'PuL',
    17: 'VP'
}

MOREL_LIST = ['AN', 'VM', 'VL', 'MGN', 'MD', 'PuA', 'LP',
              'IL', 'VA', 'Po', 'LGN', 'PuM', 'PuI', 'PuL', 'VP']
