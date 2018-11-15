"""
Execute SEER on all cores
"""
from __future__ import print_function, division

import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from data.handler import get_all_projects
from SEER import seer
import multiprocessing
from pdb import set_trace
from utils import brew_pickle, dump_json


def get_source_target():
    """
    This method performs HDP.
    :return:
    """
    all_projects = get_all_projects()  # Get a dictionary of all projects and their respective pathnames.
    project_pairs = []
    count = 0  # I use this for referencing saved pickle files
    for target in all_projects.keys():
        for source in all_projects.keys():
            if not source == target:
                count += 1
                project_pairs.append((all_projects[source], all_projects[target], count))

    return project_pairs


def pool_test(argumet):
    source, target = argumet
    return (source, target)


def execute(project_pairs):
    """
    This method performs HDP in parallel
    :return:
    """
    source, target, count = project_pairs
    result = seer(source, target)
    dump_json(result,  dir='pickles_downsamp', fname=str(count))


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=12)
    project_pairs = get_source_target()
    pool.map(execute, project_pairs)
