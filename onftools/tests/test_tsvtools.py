""" Tests for tsvtools module
"""

from os import unlink as remove
from os.path import (join as pjoin, exists, dirname, split as psplit, relpath)
from glob import glob
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from onftools.tsvtools import (parse_tsv_name, tsv2events, three_column,
                               write_tasks)
from onftools.check3col import check_task, older_cond_filenames

import pytest

HERE = dirname(__file__)
NEW_COND_PATH = pjoin(HERE, 'data', 'ds009')
OLD_COND_PATH = NEW_COND_PATH

# Parsing for ds009 tasks

def bart_processor(df):
    """ Process dataframe for BART trial types """
    before_explode = np.zeros(len(df), dtype=bool)
    action = df['action']
    explodes = action == 'EXPLODE'
    before_explode[:-1] = explodes[1:]
    trial_type = action.apply(str.lower)
    trial_type[trial_type == 'accept'] = 'inflate'
    trial_type[before_explode] = 'beforeexplode'
    duration = df['reaction_time'].copy()
    duration[explodes] = 2
    amplitude = pd.Series(np.ones(len(df)))
    classified = pd.concat([trial_type, df['onset'], duration, amplitude],
                           axis=1)
    classified.columns = ['trial_type', 'onset', 'duration', 'amplitude']
    return classified


def ss_processor(df):
    """ Process dataframe for SS trial types """
    trial_type_orig, onset, duration, arrow, button, button_code = [
        df[name] for name in
        ['trial_type', 'onset', 'duration',
         'PresentedStimulusArrowDirection',
         'SubjectResponseButton',
         'SubjectResponseButtonCode']]
    trial_type = trial_type_orig.copy()
    trial_type[(trial_type_orig == 'GO') & (arrow == button)] = 'gocorrect'
    trial_type[(trial_type == "STOP") & (button_code == 0)] = 'stopcorrect'
    trial_type[(trial_type == "STOP") & (button_code != 0)] = 'stopincorrect'
    trial_type[(trial_type_orig == 'GO') & (arrow != button)] = 'goincorrect'
    amplitude = pd.Series(np.ones(len(df)), name='amplitude')
    return pd.concat([trial_type, onset, duration, amplitude], axis=1)


# Code goes from button number to value on rating scale.  Reverse engineered
# from the original rating_par_orig field.
# Pandas 0.20.3 loads this column as object.  0.23.4 seems to load as float.
ER_RESPONSE_MAP_STR = {'114': 4, '103': 3, '121': 2, '98': 1, '0': 0,
                       'n/a': np.nan}
ER_RESPONSE_MAP_FLOAT = {114: 4, 103: 3, 121: 2, 98: 1, 0: 0, np.nan: np.nan}

def er_processor(df):
    """ Process dataframe for ER trial types """
    onset, duration, trial_type, image_type, response, rt = [
        df[name].copy() for name in
        ['onset', 'duration', 'trial_type', 'image_type',
         'response', 'reaction_time']]
    # Recode the response values using the map above.
    er_response_map = (ER_RESPONSE_MAP_STR if response.dtype == np.object else
                       ER_RESPONSE_MAP_FLOAT)
    response = response.map(er_response_map)
    tt = trial_type.copy()  # A pandas series
    tt[(trial_type == 'attend') & (image_type == 'negative')] = 'attendneg'
    tt[(trial_type == 'attend') & (image_type == 'neutral')] = 'attendneu'
    tt[(trial_type == 'suppress') & (image_type == 'negative')] = 'suppressneg'
    assert not any((trial_type == 'suppress') & (image_type == 'neutral'))
    tt[(trial_type == "rate") & (response == 0)] = 'ratemiss'
    tt[(trial_type == "rate") & (response != 0)] = 'rate'
    # Use RTs as durations for rate
    good_rates = tt == 'rate'
    # Foo
    duration[good_rates] = pd.to_numeric(rt[good_rates])
    # Make main set of events (excluding parametric regressor)
    amplitude = pd.Series(np.ones(len(df)), name='amplitude')
    main_trials = pd.concat([tt, onset, duration, amplitude], axis=1)
    # Add parametric trial type
    good_onsets = onset[good_rates]
    good_durations = duration[good_rates]
    good_responses = response[good_rates]
    amp_extra = good_responses - np.mean(good_responses)
    amp_extra.name = 'amplitude'
    tt_extra = tt[good_rates]
    tt_extra[:] = 'ratepar'
    # Put the new trials at the end
    extra = pd.concat([tt_extra, good_onsets, good_durations, amp_extra],
                      axis=1)
    return pd.concat([main_trials, extra], axis=0, ignore_index=True)


TD_RESPONSE_MAP = {'b': 0, 'y': 1, 'n/a': np.nan}

def td_processor(df):
    """ Process dataframe for TD trial types """
    onset_new, duration_orig, trial_type, delay, response, rt, onset_orig = [
        df[name].copy() for name in
        ['onset', 'duration', 'trial_type',
         'delay_time_days',
         'response_button',
         'reaction_time',
         'onset_orig']]
    # onset_orig has more decimal places, use that instead.  Rounding might in
    # fact be truncation, hence the 0.1 slippage here.
    assert np.max(np.abs(onset_new - onset_orig.round(2))) <= 0.1
    onset_orig.name = 'onset'
    # Use reaction times for duration, for most trials
    rt.name = 'duration'
    # Make responses into numbers
    response = response.map(TD_RESPONSE_MAP)
    tt = trial_type.copy()
    # Use reaction time for duration
    tt[trial_type == 'easy_par'] = 'easy'
    tt[trial_type == 'hard_par'] = 'hard'
    is_missed = response.isnull()
    tt[is_missed] = 'missed'
    # Use original duration (not RT) for missed trials
    rt[is_missed] = duration_orig[is_missed]
    amplitude = pd.Series(np.ones(len(df)), name='amplitude')
    main_trials = pd.concat([tt, onset_orig, rt, amplitude], axis=1)
    # Add parametric trial types
    extras = []
    for t_type in ('easy', 'hard'):
        trial_selector = tt == t_type
        amp_extra = delay[trial_selector].copy()
        amp_extra = amp_extra - amp_extra.mean()
        amp_extra.name = 'amplitude'
        tt_extra = tt[trial_selector].copy()
        tt_extra[:] = t_type + 'par'
        extras.append(pd.concat([tt_extra,
                                 onset_orig[trial_selector],
                                 rt[trial_selector],
                                 amp_extra],
                                axis=1))
    # Put the new trials at the end
    return pd.concat([main_trials] + extras, axis=0, ignore_index=True)


TASK_DEFS = dict(
    balloonanalogrisktask=dict(old_task_no=1,
                               processor=bart_processor,
                               conditions= ['inflate', 'beforeexplode',
                                            'cashout', 'explode'],
                               ok = True,  # Set False to disable processing
                               ),
    stopsignal=dict(old_task_no=2,
                    processor=ss_processor,
                    conditions=['gocorrect', 'stopcorrect',
                                'stopincorrect', 'goincorrect'],
                    ok = True,  # Set False to disable processing
                    ),
    emotionalregulation=dict(old_task_no=3,
                             processor=er_processor,
                             conditions=['attendneg', 'attendneu',
                                         'rate',
                                         'ratepar',
                                         'suppressneg',
                                         'ratemiss',
                                        ],
                             ok = True,  # Set False to disable processing
                             ),
    discounting=dict(old_task_no=4,
                     processor=td_processor,
                     conditions=['easy', 'easypar', 'hard', 'hardpar',
                                 'missed'],
                     ok = True,  # Set True to disable processing
                     )
)

# Throw away incomplete TASK_DEFS (where field 'ok' is not True).
TASK_DEFS = {name: task_def for name, task_def in TASK_DEFS.items()
             if task_def.get('ok')}


# End of ds009 definitions

def test_parse_tsv_name():
    assert (parse_tsv_name(
        'sub-01_task-stopsignal_run-01_events.tsv') ==
        (1, 'stopsignal', 1))
    assert (parse_tsv_name(
        pjoin('foo', 'bar', 'sub-01_task-stopsignal_run-01_events.tsv')) ==
        (1, 'stopsignal', 1))
    assert (parse_tsv_name(
        'sub-13_task-balloonanalogrisktask_events.tsv') ==
        (13, 'balloonanalogrisktask', None))
    assert (parse_tsv_name('sub-02_task-stopsignal_run-02_events.tsv') ==
            (2, 'stopsignal', 2))


def test_tasks():
    for task_name, added_args in (
        ('stopsignal', {}),
        ('emotionalregulation', dict(fail=False)),
        ('discounting', dict(fail=False, onset_field='onset_orig')),
    ):
        with TemporaryDirectory() as tmpdir:
            info = TASK_DEFS[task_name]
            task_defs = {task_name: info}
            write_tasks(NEW_COND_PATH, task_defs, tmpdir)
            for tsv_path in glob(pjoin(NEW_COND_PATH,
                                       'sub-*',
                                       'func',
                                       'sub*{}*.tsv'.format(task_name))):
                label_path = pjoin(tmpdir,
                                   relpath(dirname(tsv_path), NEW_COND_PATH))
                check_task(tsv_path, OLD_COND_PATH, info,
                           label_path=label_path, **added_args)


def test_older_cond_filenames():
    info = TASK_DEFS['stopsignal']
    assert older_cond_filenames(1, 1, info) == [
        pjoin('sub001', 'model', 'model001', 'onsets', 'task002_run001',
              'cond%03d.txt' % j) for j in range(1, 5)]


def test_three_column():
    cond_file = pjoin(NEW_COND_PATH, 'sub-02', 'func',
                      'sub-02_task-stopsignal_run-02_events.tsv')
    df = pd.read_table(cond_file)
    info = TASK_DEFS['stopsignal']
    df = info['processor'](df)
    oda = three_column(df, 'gocorrect')
    assert oda.shape == (90, 3)


def test_tsv2events():
    cond_file = pjoin(NEW_COND_PATH, 'sub-02', 'func',
                      'sub-02_task-stopsignal_run-02_events.tsv')
    events = tsv2events(cond_file, TASK_DEFS['stopsignal'])
    assert len(events) == 4
    assert sorted(events) == ['gocorrect', 'goincorrect', 'stopcorrect',
                              'stopincorrect']


def test_write_tasks():
    with TemporaryDirectory() as out_dir:
        for to_write in (['stopsignal'], ['stopsignal', 'discounting']):
            # Write the files again
            defs = {k: v for k, v in TASK_DEFS.items() if k in to_write}
            write_tasks(NEW_COND_PATH, defs, out_dir)
            # Check they are the same as the original run, or missing.
            for path in glob(pjoin(NEW_COND_PATH, 'sub-*', 'func', '*.txt')):
                _, fname = psplit(path)
                made_path = pjoin(out_dir, fname)
                sub_no, task_name, run_no = parse_tsv_name(path)
                if task_name not in to_write:
                    assert not exists(made_path)
                continue
                with open(path, 'rt') as fobj:
                    original_contents = fobj.read()
                with open(made_path, 'rt') as fobj:
                    made_contents = fobj.read()
                assert original_contents == made_contents
                remove(made_path)


def test_write_tasks_errors():
    # Check write_tasks gives errors in some cases
    defs = {k: v for k, v in TASK_DEFS.items()
            if k in ['stopsignal', 'discounting']}
    with pytest.raises(ValueError):
        # When no tsv files found
        write_tasks(HERE, defs)
    defs = {'foo': defs['stopsignal']}
    with pytest.raises(ValueError):
        # When no matching tsv files found
        write_tasks(NEW_COND_PATH, defs)
