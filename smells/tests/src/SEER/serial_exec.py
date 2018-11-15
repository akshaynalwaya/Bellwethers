from __future__ import print_function, division

import os
import sys

root = os.path.join(os.getcwd().split('src')[0], 'src')
if root not in sys.path:
    sys.path.append(root)

from data.handler import get_all_projects
from SEER import seer


def execute():
    """
    This method performs HDP.
    :return:
    """
    all_projects = get_all_projects()  # Get a dictionary of all projects and their respective pathnames.
    project_pairs = []
    result = {}  # Store results here

    for target in all_projects.keys():
        for source in all_projects.keys():
            if not source == target:
                print("Target Community: {} | Source Community: {}".format(target, source))
                seer(all_projects[source], all_projects[target])


if __name__ == "__main__":
    execute()
