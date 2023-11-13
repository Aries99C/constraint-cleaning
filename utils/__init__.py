import os


def project_root():
    """
    获取项目根目录
    """
    root = os.path.realpath(os.curdir)
    while True:
        if 'utils' in os.listdir(root):
            return root
        else:
            root = os.path.dirname(root)
