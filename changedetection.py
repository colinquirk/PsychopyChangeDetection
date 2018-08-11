"""A basic change detection experiment.

Author - Colin Quirk (cquirk@uchicago.edu)

Repo: https://github.com/colinquirk/PsychopyChangeDetection

This is a common working memory paradigm used to provide a measure of K (working memory capacity).
This code can either be used before other tasks to provide a seperate measure of K or it can be
inherited and extended. If this file is run directly the defaults at the top of the page will be
used. To make simple changes, you can adjust any of these files. For more in depth changes you
will need to overwrite the methods yourself.

Note: this code relies on my templateexperiments module. You can get it from
https://github.com/colinquirk/templateexperiments and either put it in the same folder as this
code or give the path to psychopy in the preferences.

Classes:
Ktask -- The class that runs the experiment.
    See 'print Ktask.__doc__' for simple class docs or help(Ktask) for everything.
"""

from __future__ import division
from __future__ import print_function

import os
import sys
import errno

import math
import random
import itertools

import numpy as np
import psychopy

import templateexperiments as template

# Things you probably want to change
number_of_trials_per_block = 10
number_of_blocks = 2
percent_same = 0.5  # between 0 and 1
set_sizes = [6]
stim_size = 1.5  # visual degrees, used for X and Y

single_probe = True  # False to display all stimuli at test
repeat_stim_colors = False  # False to make all stimuli colors unique
repeat_test_colors = False  # False to make test colors unique from stim colors

keys = ['s', 'd']  # first is same
distance_to_monitor = 90

instruct_text = [
    ('Welcome to the experiment. Press space to begin.'),
    ('In this experiment you will be remembering colors.\n\n'
     'Each trial will start with a fixation cross. '
     'Do your best to keep your eyes on it.\n\n'
     'Then, 6 squares with different colors will appear. '
     'Remember as many colors as you can.\n\n'
     'After a short delay, the squares will reappear.\n\n'
     'If they all have the SAME color, press the "S" key. '
     'If any of the colors are DIFFERENT, press the "D" key.\n'
     'If you are not sure, just take your best guess.\n\n'
     'You will get breaks in between blocks.\n\n'
     'Press space to start.'),
]

data_directory = os.path.join(
    os.path.expanduser('~'), 'Desktop', 'ChangeDetection', 'Data')

# Things you probably don't need to change, but can if you want to
exp_name = 'ChangeDetection'

iti_time = 1
sample_time = .25
delay_time = 1

allowed_deg_from_fix = 15

# minimum euclidean distance of stimuli in precent of allowed space
min_distance = 0.1
max_per_quad = 2  # int or None for totally random displays

colors = [
    [1, -1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [1,  1, -1],
    [1, -1,  1],
    [-1,  1,  1],
    [1,  1,  1],
    [-1, -1, -1],
    [1,  0, -1],
]

data_fields = [
    'Block',
    'Trial',
    'Timestamp',
    'Condition',
    'SetSize',
    'RT',
    'CRESP',
    'RESP',
    'ACC',
    'LocationTested',
    'Locations',
    'SampleColors',
    'TestColors',
]

gender_options = [
    'Male',
    'Female',
    'Other/Choose Not To Respond',
]

hispanic_options = [
    'Yes, Hispanic or Latino/a',
    'No, not Hispanic or Latino/a',
    'Choose Not To Respond',
]

race_options = [
    'American Indian or Alaskan Native',
    'Asian',
    'Pacific Islander',
    'Black or African American',
    'White / Caucasian',
    'More Than One Race',
    'Choose Not To Respond',
]

# Add additional questions here
questionaire_dict = {
    'Age': 0,
    'Gender': gender_options,
    'Hispanic:': hispanic_options,
    'Race': race_options,
}


# This is the logic that runs the experiment
# Change anything below this comment at your own risk
class Ktask(template.BaseExperiment):
    """The class that runs the change detection experiment.

    Parameters:
    allowed_deg_from_fix -- The maximum distance in visual degrees the stimuli can appear from
        fixation
    colors -- The list of colors (list of 3 values, -1 to 1) to be used in the experiment.
    data_directory -- Where the data should be saved.
    delay_time -- The number of seconds between the stimuli display and test.
    instruct_text -- The text to be displayed to the participant at the beginning of the
        experiment.
    iti_time -- The number of seconds in between a response and the next trial.
    keys -- The keys to be used for making a response. First is used for 'same' and the second is
        used for 'different'
    max_per_quad -- The number of stimuli allowed in each quadrant. If None, displays are
        completely random.
    min_distance -- The minimum distance in visual degrees between stimuli.
    number_of_blocks -- The number of blocks in the experiment.
    number_of_trials_per_block -- The number of trials within each block.
    percent_same -- A float between 0 and 1 (inclusive) describing the likelihood of a trial being
        a "same" trial.
    repeat_stim_colors -- If True, a stimuli display can have repeated colors.
    repeat_test_colors -- If True, on a change trial the foil color can be one of the other colors
        from the initial display.
    sample_time -- The number of seconds the stimuli are on the screen for.
    set_sizes -- A list of all the set sizes. An equal number of trials will be shown for each set
        size.
    single_probe -- If True, the test display will show only a single probe. If False, all the
        stimuli will be shown.
    stim_size -- The size of the stimuli in visual angle.

    Additional keyword arguments are sent to template.BaseExperiment().

    Methods:
    chdir -- Changes the directory to where the data will be saved.
    display_break -- Displays a screen during the break between blocks.
    display_fixation -- Displays a fixation cross.
    display_stimuli -- Displays the stimuli.
    display_test -- Displays the test array.
    get_response -- Waits for a response from the participant.
    make_block -- Creates a block of trials to be run.
    make_trial -- Creates a single trial.
    questionaire_dict -- Questions to be included in the dialog.
    run_trial -- Runs a single trial.
    run -- Runs the entire experiment.
    """

    def __init__(self, number_of_trials_per_block=number_of_trials_per_block,
                 number_of_blocks=number_of_blocks, percent_same=percent_same,
                 set_sizes=set_sizes, stim_size=stim_size, colors=colors,
                 keys=keys, allowed_deg_from_fix=allowed_deg_from_fix,
                 min_distance=min_distance, max_per_quad=max_per_quad,
                 instruct_text=instruct_text, single_probe=single_probe,
                 iti_time=iti_time, sample_time=sample_time,
                 delay_time=delay_time, repeat_stim_colors=repeat_stim_colors,
                 repeat_test_colors=repeat_test_colors, data_directory=data_directory,
                 questionaire_dict=questionaire_dict, **kwargs):

        self.number_of_trials_per_block = number_of_trials_per_block
        self.number_of_blocks = number_of_blocks
        self.percent_same = percent_same
        self.set_sizes = set_sizes
        self.stim_size = stim_size

        self.colors = colors

        self.iti_time = iti_time
        self.sample_time = sample_time
        self.delay_time = delay_time

        self.keys = keys

        self.allowed_deg_from_fix = allowed_deg_from_fix

        self.min_distance = min_distance

        if max_per_quad is not None and max(self.set_sizes)/4 > max_per_quad:
            raise ValueError('Max per quad is too small.')

        self.max_per_quad = max_per_quad

        self.data_directory = data_directory
        self.instruct_text = instruct_text
        self.questionaire_dict = questionaire_dict

        self.single_probe = single_probe
        self.repeat_stim_colors = repeat_stim_colors
        self.repeat_test_colors = repeat_test_colors

        self.same_trials_per_set_size = int((
            number_of_trials_per_block / len(set_sizes)) * percent_same)

        if self.same_trials_per_set_size % 1 != 0:
            raise ValueError('Each condition needs a whole number of trials.')
        else:
            self.diff_trials_per_set_size = (
                number_of_trials_per_block - self.same_trials_per_set_size)

        super(Ktask, self).__init__(**kwargs)

    def chdir(self):
        """Changes the directory to where the data will be saved.
        """

        try:
            os.makedirs(self.data_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        os.chdir(self.data_directory)

    def make_block(self):
        """Makes a block of trials.

        Returns a shuffled list of trials created by self.make_trial.
        """

        trial_list = []

        self.same_trials_per_set_size

        for set_size in self.set_sizes:
            for _ in range(self.same_trials_per_set_size):
                trial = self.make_trial(set_size, 'same')
                trial_list.append(trial)

            for _ in range(self.diff_trials_per_set_size):
                trial = self.make_trial(set_size, 'diff')
                trial_list.append(trial)

        random.shuffle(trial_list)

        return trial_list

    def _make_grid(self):
        """A private method that creates a grid of possible locations. A helper function for
            self.generate_locations.

        Returns a list of possible locations.

        Parameters:
        set_size -- The number of stimuli for this trial.
        """
        grid_dist = self.min_distance * 2
        grid_jitter = random.uniform(0, grid_dist)
        num_lines = int(math.floor(1 / (grid_dist)))
        center = np.array([.5, .5])
        grid = []
        for x in range(num_lines):
            for y in range(num_lines):
                loc = [(grid_dist * x) + grid_jitter,
                       (grid_dist * y) + grid_jitter]

                dist_to_center = np.linalg.norm(np.array(loc) - center)
                too_big = loc[0] >= 1 or loc[1] >= 1

                # Add it if it's not too close to the center or outside range
                if not dist_to_center < grid_dist and not too_big:
                    grid.append(loc)

        return grid

    def generate_locations(self, set_size):
        """Creates the locations for a trial. A helper function for self.make_trial.

        Returns a list of acceptable locations.

        Parameters:
        set_size -- The number of stimuli for this trial.
        """
        if self.max_per_quad is not None:
            # quad boundries (x1, x2, y1, y2)
            quad_count = [0, 0, 0, 0]

        grid = self._make_grid()

        locs = []
        while len(locs) <= set_size:
            grid_attempt = random.choice(grid)
            attempt = [coord + random.uniform(-self.min_distance / 2, self.min_distance / 2)
                       for coord in grid_attempt]

            if self.max_per_quad is not None:
                if attempt[0] < 0.5 and attempt[1] < 0.5:
                    quad = 0
                elif attempt[0] >= 0.5 and attempt[1] < 0.5:
                    quad = 1
                elif attempt[0] < 0.5 and attempt[1] >= 0.5:
                    quad = 2
                else:
                    quad = 3

                if quad_count[quad] < self.max_per_quad:
                    quad_count[quad] += 1
                    grid.remove(grid_attempt)
                    locs.append(attempt)
            else:
                grid.remove(grid_attempt)
                locs.append(attempt)

        return locs

    def make_trial(self, set_size, condition):
        """Creates a single trial dict. A helper function for self.make_block.

        Returns the trial dict.

        Parameters:
        set_size -- The number of stimuli for this trial.
        condition -- Whether this trial is same or different.
        """

        if condition == 'same':
            cresp = self.keys[0]
        else:
            cresp = self.keys[1]

        test_location = random.choice(range(set_size))

        if self.repeat_stim_colors:
            stim_colors = [random.choice(self.colors) for _ in range(set_size)]
        else:
            stim_colors = random.sample(self.colors, set_size)

        if self.repeat_test_colors:
            test_color = random.choice(self.colors)
            while test_color == self.colors[test_location]:
                test_color = random.choice(self.colors)
        else:
            test_color = random.choice(
                [color for color in self.colors if color not in stim_colors])

        locs = self.generate_locations(set_size)

        trial = {
            'set_size': set_size,
            'condition': condition,
            'cresp': cresp,
            'locations': locs,
            'stim_colors': stim_colors,
            'test_color': test_color,
            'test_location': test_location,
        }

        return trial

    def display_break(self):
        """Displays a break screen in between blocks.
        """

        break_text = 'Please take a short break. Press space to continue.'
        self.display_text_screen(text=break_text, bg_color=[204, 255, 204])

    def display_fixation(self, wait_time):
        """Displays a fixation cross. A helper function for self.run_trial.

        Parameters:
        wait_time -- The amount of time the fixation should be displayed for.
        """

        psychopy.visual.TextStim(
            self.experiment_window, text='+', color=[-1, -1, -1]).draw()
        self.experiment_window.flip()

        psychopy.core.wait(wait_time)

    def display_stimuli(self, coordinates, colors):
        """Displays the stimuli. A helper function for self.run_trial.

        Parameters:
        coordinates -- A list of coordinates (list of x and y value) describing where the stimuli
            should be displayed.
        colors -- A list of colors describing what should be drawn at each coordinate.
        """

        psychopy.visual.TextStim(
            self.experiment_window, text='+', color=[-1, -1, -1]).draw()

        for pos, color in itertools.izip(coordinates, colors):
            psychopy.visual.Rect(
                self.experiment_window, height=self.stim_size,
                width=self.stim_size, pos=pos, fillColor=color,
                units='deg').draw()

        self.experiment_window.flip()

        psychopy.core.wait(self.sample_time)

    def display_test(self, condition, coordinates, colors, test_loc, test_color):
        """Displays the test array. A helper function for self.run_trial.

        Parameters:
        condition -- Whether the trial is same or different.
        coordinates -- A list of coordinates where stimuli should be drawn.
        colors -- The colors that should be drawn at each coordinate.
        test_loc -- The index of the tested stimuli.
        test_color -- The color of the tested stimuli.
        """

        psychopy.visual.TextStim(
            self.experiment_window, text='+', color=[-1, -1, -1]).draw()

        if self.single_probe:
            psychopy.visual.Rect(
                self.experiment_window, width=self.stim_size,
                height=self.stim_size, pos=coordinates[test_loc],
                fillColor=colors[test_loc], units='deg').draw()

        else:
            for pos, color in itertools.izip(coordinates, colors):
                psychopy.visual.Rect(
                    self.experiment_window, width=self.stim_size,
                    height=self.stim_size, pos=pos, fillColor=color,
                    units='deg').draw()

        # Draw over the test color on diff trials
        if condition == 'diff':
            psychopy.visual.Rect(
                self.experiment_window, width=self.stim_size,
                height=self.stim_size, pos=coordinates[test_loc],
                fillColor=test_color, units='deg').draw()

        self.experiment_window.flip()

        psychopy.core.wait(self.sample_time)

    def get_response(self):
        """Waits for a response from the participant. A helper function for self.run_trial.

        Pressing Q while the function is wait for a response will quit the experiment.

        Returns the pressed key and the reaction time.
        """

        rt_timer = psychopy.core.MonotonicClock()

        keys = self.keys + ['q']

        resp = psychopy.event.waitKeys(keyList=keys, timeStamped=rt_timer)

        if 'q' in resp[0]:
            self.quit_experiment()

        return resp[0][0], resp[0][1]*1000  # key and rt in milliseconds

    def send_data(self, data):
        self.update_experiment_data([data])

    def run_trial(self, trial, block_num, trial_num):
        """Runs a single trial.

        Parameters:
        trial -- The dictionary of information about a trial.
        block_num -- The number of the block in the experiment.
        trial_num -- The number of the trial within a block.
        """

        coordinates = [[(num - .5) * self.allowed_deg_from_fix for num in loc]
                       for loc in trial['locations']]

        self.display_fixation(self.iti_time)
        self.display_stimuli(coordinates, trial['stim_colors'])
        self.display_fixation(self.delay_time)
        self.display_test(
            trial['condition'], coordinates, trial['stim_colors'],
            trial['test_location'], trial['test_color'])

        resp, rt = self.get_response()

        acc = 1 if resp == trial['cresp'] else 0

        data = {
            'Block': block_num,
            'Trial': trial_num,
            'Timestamp': psychopy.core.getAbsTime(),
            'Condition': trial['condition'],
            'SetSize': trial['set_size'],
            'RT': rt,
            'CRESP': trial['cresp'],
            'RESP': resp,
            'ACC': acc,
            'LocationTested': trial['test_location'],
            'Locations': trial['locations'],
            'SampleColors': trial['stim_colors'],
            'TestColor': trial['test_color'],
        }

        return data

    def run(self):
        """Runs the entire experiment if the file is run directly.
        """

        self.chdir(self)

        ok = self.get_experiment_info_from_dialog(self.questionaire_dict)

        if not ok:
            print('Experiment has been terminated.')
            sys.exit(1)

        self.save_experiment_info()
        self.open_csv_data_file()
        self.open_window(screen=0)
        self.display_text_screen('Loading...', wait_for_input=False)

        for instruction in self.instruct_text:
            self.display_text_screen(text=instruction)

        for block_num in range(self.number_of_blocks):
            block = self.make_block()
            for trial_num, trial in enumerate(block):
                data = self.run_trial(trial, block_num, trial_num)
                self.send_data(data)

            self.save_data_to_csv()

            if block_num + 1 != self.number_of_blocks:
                self.display_break()

        self.display_text_screen(
            'The experiment is now over, please get your experimenter.',
            bg_color=[0, 0, 255], text_color=[255, 255, 255])

        self.quit_experiment()


# If you call this script directly, the task will run with your defaults
if __name__ == '__main__':
    exp = Ktask(
        # BaseExperiment parameters
        experiment_name=exp_name,
        data_fields=data_fields,
        monitor_distance=distance_to_monitor,
        # Custom parameters go here
    )

    exp.run()
