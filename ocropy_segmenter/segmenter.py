#!/usr/bin/env python
import os
import os.path
import sys

import cv2
import numpy as np
import scipy.ndimage as ndimage
from imageio import imwrite

from ocropy_segmenter import misc_components as pg_seg_aux


class DefaultArgs:
    """Just an object to add attributes to emulate the arg_parser object"""

    minscale = 12.0  # 'minimum scale permitted'
    scale = 0.0  # 'the basic scale of the document (roughly, xheight) 0=automatic'
    hscale = 1.0  # 'non-standard scaling of horizontal parameters'
    vscale = 1.0  # 'non-standard scaling of vertical parameters'
    threshold = 0.2  # 'baseline threshold'
    noise = 8  # 'noise threshold for removing small components from lines'
    usegauss = True  # 'use gaussian instead of uniform'
    pad = 3  # 'padding for extracted lines'
    expand = 3  # 'expand mask for grayscale extraction'
    output = None  # the output path
    output_file = None  # the actual name of the output file
    output_suffix = ''
    output_segs = False
    maxseps = 0  # 'maximum black column separators'
    sepwiden = 10  # 'widen black separators (to account for warping)'
    maxcolseps = 3  # 'maximum # whitespace column separators'
    csminheight = 10  # 'minimum column height (units=scale)
    file = None
    gss_blr_sx = 1
    gss_blr_sy = 0
    bin_thresh = 101
    extra_erode = False
    hline_perc = 0.10
    vline_perc = 0.15


def main_process(file, kwargs):

    def print_error(*objs):
        print("ERROR: ", *objs, file=sys.stderr)

    # ###############################################################
    # ## Column finding.
    # ##
    # ## This attempts to find column separators, either as extended
    # ## vertical black lines or extended vertical whitespace.
    # ## It will work fairly well in simple cases, but for unusual
    # ## documents, you need to tune the parameters or use a mask.
    # ###############################################################

    def compute_separators_morph(binary, scale, args):
        """Finds vertical black lines corresponding to column separators."""
        d0 = int(max(5, scale / 4))
        d1 = int(max(5, scale)) + args.sepwiden
        thick = pg_seg_aux.r_dilation(binary, (d0, d1))
        vert = pg_seg_aux.rb_opening(thick, (10 * scale, 1))
        vert = pg_seg_aux.r_erosion(vert, (d0 // 2, args.sepwiden))
        vert = pg_seg_aux.select_regions(vert, pg_seg_aux.dim1, min=3, nbest=2 * args.maxseps)
        vert = pg_seg_aux.select_regions(vert, pg_seg_aux.dim0, min=20 * scale, nbest=args.maxseps)
        return vert

    def compute_colseps_conv(binary, args, scale=1.0):
        """Find column separators by convolution and
        thresholding."""
        # find vertical whitespace by thresholding
        smoothed =  ndimage.gaussian_filter(1.0 * binary, (scale, scale * 0.5))
        smoothed = ndimage.uniform_filter(smoothed, (5.0 * scale, 1))
        thresh = (smoothed < np.amax(smoothed) * 0.1)
        # find column edges by filtering
        grad =  ndimage.gaussian_filter(1.0 * binary, (scale, scale * 0.5), order=(0, 1))
        grad =  ndimage.uniform_filter(grad, (10.0 * scale, 1))
        grad = (grad > 0.5 * np.amax(grad))
        # combine edges and whitespace
        seps = np.minimum(thresh,  ndimage.maximum_filter(grad, (int(scale), int(5 * scale))))
        seps =  ndimage.maximum_filter(seps, (int(2 * scale), 1))
        # select only the biggest column separators
        seps = pg_seg_aux.select_regions(seps, pg_seg_aux.dim0, min=args.csminheight * scale, nbest=args.maxcolseps)
        return seps

    def compute_colseps(binary, scale, args):
        """Computes column separators either from vertical black lines or whitespace."""
        # print_info("considering at most %g whitespace column separators" % args.maxcolseps)
        colseps = compute_colseps_conv(binary=binary, args=args, scale=scale)
        if args.maxseps > 0:
            # print_info("considering at most %g black column separators" % args.maxseps)
            seps = compute_separators_morph(binary, scale, args)
            colseps = np.maximum(colseps, seps)
            binary = np.minimum(binary, 1 - seps)
        return colseps, binary



    ################################################################
    ### Text Line Finding.
    ###
    ### This identifies the tops and bottoms of text lines by
    ### computing gradients and performing some adaptive thresholding.
    ### Those components are then used as seeds for the text lines.
    ################################################################

    def compute_gradmaps(binary, scale, args):
        # use gradient filtering to find baselines
        boxmap = pg_seg_aux.compute_boxmap(binary, scale)
        cleaned = boxmap * binary

        if args.usegauss:
            # this uses Gaussians
            grad = ndimage.gaussian_filter(1.0 * cleaned, (args.vscale * 0.3 * scale,
                                                           args.hscale * 6 * scale), order=(1, 0))
        else:
            # this uses non-Gaussian oriented filters
            grad = ndimage.gaussian_filter(1.0 * cleaned, (max(4, args.vscale * 0.3 * scale),
                                                           args.hscale * scale), order=(1, 0))
            grad = ndimage.uniform_filter(grad, (args.vscale, args.hscale * 6 * scale))

        bottom = pg_seg_aux.norm_max((grad < 0) * (-grad))
        top = pg_seg_aux.norm_max((grad > 0) * grad)
        return bottom, top, boxmap

    def compute_line_seeds(binary, bottom, top, colseps, scale, args):
        """Base on gradient maps, computes candidates for baselines
        and xheights.  Then, it marks the regions between the two
        as a line seed."""
        t = args.threshold
        vrange = int(args.vscale * scale)
        bmarked = ndimage.maximum_filter(bottom ==  ndimage.maximum_filter(bottom, (vrange, 0)), (2, 2))
        bmarked = bmarked * (bottom > t * np.amax(bottom) * t) * (1 - colseps)
        tmarked = ndimage.maximum_filter(top == ndimage.maximum_filter(top, (vrange, 0)), (2, 2))
        tmarked = tmarked * (top > t * np.amax(top) * t / 2) * (1 - colseps)
        tmarked = ndimage.maximum_filter(tmarked, (1, 20))
        seeds = np.zeros(binary.shape, 'i')
        delta = max(3, int(scale / 2))
        for x in range(bmarked.shape[1]):
            transitions = sorted([(y, 1) for y in pg_seg_aux.find(bmarked[:, x])] + [(y, 0) for y in pg_seg_aux.find(tmarked[:, x])])[::-1]
            transitions += [(0, 0)]
            for l in range(len(transitions) - 1):
                y0, s0 = transitions[l]
                if s0 == 0: continue
                seeds[y0 - delta:y0, x] = 1
                y1, s1 = transitions[l + 1]
                if s1 == 0 and (y0 - y1) < 5 * scale: seeds[y1:y0, x] = 1
        seeds =  ndimage.maximum_filter(seeds, (1, int(1 + scale)))
        seeds = seeds * (1 - colseps)
        seeds, _ = ndimage.label(seeds)
        return seeds

    ################################################################
    ### The complete line segmentation process.
    ################################################################

    def compute_segmentation(binary, scale, args):
        """Given a binary image, compute a complete segmentation into
        lines, computing both columns and text lines."""
        binary = np.array(binary, 'B')
        binary = pg_seg_aux.erode_hlines_and_vlines(binary, scale, args)
        # do the column finding
        colseps, binary = compute_colseps(binary, scale, args)
        # now compute the text line seeds
        bottom, top, boxmap = compute_gradmaps(binary, scale, args)
        seeds = compute_line_seeds(binary, bottom, top, colseps, scale, args)
        # spread the text line seeds to all the remaining components
        llabels = pg_seg_aux.propagate_labels(boxmap, seeds, conflict=0)
        spread = pg_seg_aux.spread_labels(seeds, maxdist=scale)
        llabels = np.where(llabels > 0, llabels, spread * binary)
        segmentation = llabels * binary
        return segmentation

    ################################################################
    ### Process the file.
    ################################################################

    args = DefaultArgs()
    args.file = file
    args.__dict__.update(**kwargs)
    assert args.file is not None

    # if the file is a string object then attempt to load
    if isinstance(args.file, str):
        try:
            binary = pg_seg_aux.read_image_binary(args.file)
        except IOError:
            print_error(f'Cannot open file: \n{args.file}')
            return
    elif isinstance(args.file, np.ndarray):
        pg_seg_aux.check_binary(args.file)
        binary = args.file
    else:
        raise TypeError("File object must either be a string path to a .png file, or an ndarray object")

    if args.output is not None:
        try:
            assert args.output_file is not None
        except AssertionError:
            raise AttributeError('Argument \'output_file\' must be specified if output path is sepcified.')
        png_name = os.path.splitext(args.output_file)[0]
        args.output = os.path.join(args.output, png_name + args.output_suffix)
        if not os.path.exists(args.output):
            os.mkdir(args.output)

    binary = 1 - binary  # invert
    if args.scale == 0:
        scale = pg_seg_aux.estimate_scale(binary)
    else:
        scale = args.scale
    # find columns and text lines
    segmentation = compute_segmentation(binary=binary, scale=scale, args=args)

    # compute the reading order
    lines = pg_seg_aux.compute_lines(segmentation, args.minscale)
    order = pg_seg_aux.reading_order([l.bounds for l in lines])
    lsort = pg_seg_aux.topsort(order)

    # renumber the labels so that they conform to the specs
    nlabels = np.amax(segmentation) + 1
    renumber = np.zeros(nlabels, 'i')
    for i, v in enumerate(lsort):
        renumber[lines[v].label] = 0x010000 + (i + 1)
    segmentation = renumber[segmentation]

    # finally, output everything
    lines = [lines[i] for i in lsort]

    if args.output_segs:
        pg_seg_aux.write_page_segmentation(f"{png_name}.pseg.png", segmentation)

    binlines = []
    for i, l in enumerate(lines):
        binlines.append(pg_seg_aux.extract_masked((1- segmentation), l, pad=args.pad, expand=args.expand))
        if args.output is not None:
            pg_seg_aux.write_image_binary(os.path.join(args.output, f'{png_name}_seg_{i + 1}.png'), binlines[-1])

    # if output folder is not set then return the list of segment pngs
    if args.output is None:
        return binlines




