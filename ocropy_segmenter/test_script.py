import os

import cv2

from ocropy_segmenter import segmenter
import multiprocessing as mp


def main_root_path():
    _bp = os.getcwd()
    for i in range(4):
        _bp = os.path.split(_bp)[0]
    assert os.path.exists(_bp)
    _bp = os.path.join(_bp, 'NSQIP_Abstraction_Automation_Master', 'data_files', 'scanned_forms',
                       'seg_test', 'pre_op_checklist_p2')
    return _bp

def find_new_files(src_path, dst_path, ftype, reprocess):
    """
    Find all files of ftype in the src_path that are not in the dst_path
    :param src_path: str - full folder path of source files
    :param dst_path: str - full folder path of destination files
    :param ftype: str - '.filetype' - file type to search for with '.' included
    :param reprocess: - boolean to set whether to process only new files or all ocropy_segmenter files
    :return:
    """

    # collect the png files from the ocropy_segmenter folder
    src_files = set([file for r, d, f in os.walk(src_path) for file in f if ftype in file])

    if not reprocess:
        # collect the png files from the dst folder
        dst_files = set([file for r, d, f in os.walk(dst_path) for file in f if ftype in file])
        # take only the ocropy_segmenter files not in the dst folder
        src_files = src_files - dst_files

    return [os.path.join(src_path, f) for f in src_files]


def main():

    use_mp = False
    rpath = main_root_path()
    src_scan_path = os.path.join(rpath, 'cropped')
    dest_scan_path = os.path.join(rpath, 'segged')

    file_list = find_new_files(src_path=src_scan_path, dst_path=dest_scan_path, ftype='.png', reprocess=False)
    file_dict = {os.path.splitext(os.path.split(f)[1])[0]:f for f in file_list}
    total_files = len(file_list)
    kwargs = {'minscale':6.0, 'pad': 15, 'threshold': 0.05, 'total_files':total_files}
    # kwargs = {'minscale': 6.0, 'pad': 15, 'threshold': 0.05, 'output': dest_scan_path, 'total_files': total_files}

    if use_mp:
        pool = mp.Pool(mp.cpu_count())
        pool.starmap(segmenter.main_process, [(file, kwargs, ind) for ind, (fname, file) in enumerate(file_dict.items())])
        pool.close()

    else:
        for ind, (fname, file) in enumerate(file_dict.items()):
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            img = cv2.threshold(img, 101, 255, cv2.THRESH_BINARY)[1] // 255
            segs = segmenter.main_process(img, kwargs, ind)
    print('Finished')


if __name__ == '__main__':
    main()