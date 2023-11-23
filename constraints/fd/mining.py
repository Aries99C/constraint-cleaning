from utils import project_root

PROJECT_ROOT = project_root()


def read_from_TANE(dataset, verbose=0):
    fd_list = []
    with open(PROJECT_ROOT + '/rules/{}_fd.txt'.format(dataset), 'r') as f:
        for line in f.readlines():
            lhs, rhs = line.split(',')[0].strip(), line.split(',')[1].strip()

