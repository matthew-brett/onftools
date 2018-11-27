""" Utilties for reading tsv files, writing 3-column event files.
"""

from glob import glob
from os import makedirs
from os.path import join as pjoin, split as psplit, relpath, dirname, exists

import numpy as np

import pandas as pd


def parse_tsv_name(tsv_path):
    """ Parse tsv file name, return subject no, task name, run_no

    Parameters
    ----------
    tsv_path : str
        .tsv filename.

    Returns
    -------
    subject_no : str
        E.g. "sub-12"
    task_name : str
        E.g. "stopsignal"
    run_no : None or int
        None if no run number specified, otherwise a 1-based integer giving the
        run number, where 1 is the first run.
    """
    path, fname = psplit(tsv_path)
    parts = fname.split('_')
    if len(parts) == 3:
        run_no = None
    else:
        run_parts = parts.pop(2).split('-')
        assert run_parts[0] == 'run'
        run_no = int(run_parts[1])
    sub_parts = parts[0].split('-')
    assert sub_parts[0] == 'sub'
    sub_no = int(sub_parts[1])
    task_name = parts[1].split('-')[1]
    return sub_no, task_name, run_no


def three_column(df, name):
    """ Return 3-column onset, duration, amplitude data frame for event `name`
    """
    ons_dur_amp = df[df['trial_type'] == name]
    return ons_dur_amp[['onset', 'duration', 'amplitude']].values


def tsv2events(tsv_path, task_def):
    """ Return dictionary of 3-column event dataframes from `tsv_path`

    Parameters
    ----------
    conditions : sequence
        Sequence of condition names
    task_def : dict
        Dict with key 'conditions', value a sequence of condition names.
        Optionally, key 'processor' with value: a callable to process the read
        data frame.

    Returns
    -------
    events : dict
        Keys are condition names, values are three column event array.
    """
    df = pd.read_table(tsv_path)
    if 'processor' in task_def:
        df = task_def['processor'](df)
    return {name: three_column(df, name) for name in task_def['conditions']}


def write_task(tsv_path, task_def, out_path=None):
    """ Write .txt event files for .tsv event definitions

    Parameters
    ----------
    tsv_path : str
        Path to .tsv file.
    task_def : dict
        Dict with key 'conditions', value a sequence of condition names.
        Optionally, key 'processor' with value: a callable to process the read
        data frame.
    out_path : None or str, optional
        If str, directory to write output .txt files.  If None, use directory
        containing the .tsv file in `tsv_path`.
    """
    sub_no, task_name, run_no = parse_tsv_name(tsv_path)
    events = tsv2events(tsv_path, task_def)
    if len(events) == 0:
        return
    tsv_dir, fname = psplit(tsv_path)
    path = tsv_dir if out_path is None else out_path
    run_part = '' if run_no is None else '_run-%02d' % run_no
    fname_prefix = pjoin(
        path,
        'sub-%02d_task-%s%s_label-' % (sub_no, task_name, run_part))
    for name in events:
        new_fname = fname_prefix + name + '.txt'
        oda = events[name]
        if len(oda):
            print('Writing from', tsv_path, 'to', new_fname)
            np.savetxt(new_fname, oda, '%f', '\t')


def write_tasks(start_path, task_defs, out_root=None):
    """ Write .txt event files for all tasks with defined processing.

    Parameters
    ----------
    start_path : str
        Path containing subject directories such as ``sub-01`` etc.
    task_defs : dict
        Keys are task names, values are dicts with keys 'processor' and
        'conditions'.  These are: a callable to process the read data frame,
        and the list of condition names.
    out_root : None or str, optional
        If str, root directory to write output .txt files.  Files written in
        subdirectories corresponding to subdirectories of TSV files found
        within `start_path`.  If None, use directory containing the .tsv file,
        found by searching in `start_path`.
    """
    globber = pjoin(start_path, 'sub-*', 'func', 'sub*tsv')
    matches = glob(globber)
    if len(matches) == 0:
        raise ValueError(f'No matches for glob "{globber}"')
    found = False
    out_path = None
    for tsv_path in matches:
        sub_no, task_name, run_no = parse_tsv_name(tsv_path)
        if task_name in task_defs:
            if out_root is not None:
                rel_to_start = relpath(dirname(tsv_path), start_path)
                out_path = pjoin(out_root, rel_to_start)
                if not exists(out_path):
                    makedirs(out_path)
            info = task_defs[task_name]
            write_task(tsv_path, info, out_path)
            found = True
    if not found:
        task_names = ''.join(task_defs)
        raise ValueError(f'No found files match tasks {task_names}')
