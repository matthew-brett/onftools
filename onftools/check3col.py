""" Check 3 column event files against previous versions
"""
from os.path import join as pjoin, exists, split as psplit

import numpy as np
import pandas as pd

from .tsvtools import (parse_tsv_name, tsv2events)


def older_cond_filenames(sub_no, run_no, task_def, model_no=1):
    run_no = run_no if run_no is not None else 1
    root = pjoin('sub%03d' % sub_no,
                 'model',
                 'model%03d' % model_no,
                 'onsets',
                 'task%03d_run%03d' % (task_def['old_task_no'], run_no))
    filenames = []
    for i in range(len(task_def['conditions'])):
        filenames.append(pjoin(root, 'cond%03d.txt' % (i + 1)))
    return filenames


def check_task(tsv_path,
               old_path,
               task_def,
               model_no=1,
               fail=False,
               onset_field='onset'):
    path, fname = psplit(tsv_path)
    sub_no, task_name, run_no = parse_tsv_name(tsv_path)
    run_part = '_run-%02d' % run_no if run_no is not None else ''
    cond_fnames = older_cond_filenames(sub_no, run_no, task_def, model_no)
    orig_df = pd.read_table(tsv_path)
    events = tsv2events(tsv_path, task_def)
    new_cond_prefix = pjoin(
        path, 'sub-%02d_task-%s%s_label-' %
        (sub_no, task_name, run_part))
    for i, name in enumerate(task_def['conditions']):
        ons_dur_amp = events[name]
        # Check new event file
        new_cond_fname = new_cond_prefix + name + '.txt'
        if len(ons_dur_amp) == 0:
            assert not exists(new_cond_fname)
        else:
            new_cond_res = np.loadtxt(new_cond_fname)
            assert np.allclose(ons_dur_amp, new_cond_res, atol=1e-5)
        old_cond_fname = pjoin(old_path, cond_fnames[i])
        if not exists(old_cond_fname):
            assert(len(ons_dur_amp) == 0)
            continue
        old_events = np.atleast_2d(np.loadtxt(old_cond_fname))
        run_part = '' if run_no is None else ' run {}'.format(run_no)
        msg = 'check sub {}{} condition {} (cond no {})'.format(
            sub_no, run_part, name, (i + 1))
        if len(ons_dur_amp) != len(old_events):
            print(msg)
            print_disjoint_events(ons_dur_amp, old_events, orig_df,
                                  onset_field)
            if fail:
                assert False
        elif not np.allclose(ons_dur_amp, old_events, atol=1e-4):
            print(msg)
            print('onsets / durations / amplitudes do not match '
                  'to given precision')
            if fail:
                assert False


def print_disjoint_events(new, old, data_frame, onset_field='onset'):
    new_not_old = ons_difference(new, old)
    if new_not_old:
        print('Events in new not old')
        print(difference_report(new_not_old, data_frame, onset_field))
    old_not_new = ons_difference(old, new)
    if old_not_new:
        print('Events in old not new')
        print(difference_report(old_not_new, data_frame, onset_field))


def ons_difference(first, second):
    difference = set()
    ons_2 = second[:, 0]
    for onset in first[:, 0]:
        if not np.any(np.isclose(ons_2, onset, atol=1e-4)):
            difference.add(onset)
    return difference


def difference_report(rounded_onsets, data_frame, onset_field='onset'):
    for onset in rounded_onsets:
        filtered = data_frame[
            np.isclose(data_frame[onset_field], onset, atol=1e-4)
                      ]
        assert len(filtered) == 1
        print(filtered)
