A customizable and extendable change detection experiment written in Psychopy. Designed for the desktop.

Author - Colin Quirk (cquirk@uchicago.edu)

Repo: https://github.com/colinquirk/PsychopyChangeDetection

This is a common working memory paradigm used to provide a measure of K (working memory capacity).

This code can either be used before other tasks to provide a seperate measure of K or it can be inherited and extended. If this file is run directly the defaults at the top of the page will be used. To make simple changes, you can adjust any of these files. For more in depth changes you will need to overwrite the methods yourself.

**Note**: this code relies on my templateexperiments module. You can get it from
https://github.com/colinquirk/templateexperiments and either put it in the same folder as this
code or give the path to psychopy in the preferences.



### Classes
* Ktask -- The class that runs the experiment.
    

### Parameters
* allowed_deg_from_fix -- The maximum distance in visual degrees the stimuli can appear from fixation
* colors -- The list of colors (list of 3 values, -1 to 1) to be used in the experiment.
* data_directory -- Where the data should be saved.
* delay_time -- The number of seconds between the stimuli display and test.
* instruct_text -- The text to be displayed to the participant at the beginning of the experiment.
* iti_time -- The number of seconds in between a response and the next trial.
* keys -- The keys to be used for making a response. First is used for 'same' and the second is used for 'different'
* max_per_quad -- The number of stimuli allowed in each quadrant. If None, displays are completely * random.
* min_distance -- The minimum distance in visual degrees between stimuli.
* number_of_blocks -- The number of blocks in the experiment.
* number_of_trials_per_block -- The number of trials within each block.
* percent_same -- A float between 0 and 1 (inclusive) describing the likelihood of a trial being a "same" trial.
* questionaire_dict -- Questions to be included in the dialog.
* repeat_stim_colors -- If True, a stimuli display can have repeated colors.
* repeat_test_colors -- If True, on a change trial the foil color can be one of the other colors from the initial display.
* sample_time -- The number of seconds the stimuli are on the screen for.
* set_sizes -- A list of all the set sizes. An equal number of trials will be shown for each set size.
* single_probe -- If True, the test display will show only a single probe. If False, all the stimuli will be shown.
* stim_size -- The size of the stimuli in visual angle.

Additional keyword arguments are sent to template.BaseExperiment().

### Methods
* chdir -- Changes the directory to where the data will be saved.
* display_break -- Displays a screen during the break between blocks.
* display_fixation -- Displays a fixation cross.
* display_stimuli -- Displays the stimuli.
* display_test -- Displays the test array.
* generate_locations -- Helper function that generates locations for make_trial
* get_response -- Waits for a response from the participant.
* make_block -- Creates a block of trials to be run.
* make_trial -- Creates a single trial.
* run_trial -- Runs a single trial.
* run -- Runs the entire experiment.

## Hooks

Hooks can be sent to the `run` method in order to allow for small changes to be made without having to completely rewrite the run method in a subclass.

#### Available Hooks

- setup_hook -- takes self, executed once the window is open.
- before_first_trial_hook -- takes self, executed after instructions are displayed.
- pre_block_hook -- takes self and the block list, executed immediately before block start.
    Can optionally return an altered block list.
- pre_trial_hook -- takes self and the trial dict, executed immediately before trial start.
    Can optionally return an altered trial dict.
- post_trial_hook -- takes self and the trial data, executed immediately after trial end.
    Can optionally return altered trial data to be stored.
- post_block_hook -- takes self, executed at end of block before break screen (including
    last block).
- end_experiment_hook -- takes self, executed immediately before end experiment screen.

#### Hook Example

For example, if you wanted a block of practice trials, you could simply define a function:

```
# Arbitrary function name
def my_before_first_trial_hook(self):
    # Self refers to the experiment object
    self.display_text_screen('We will now do a practice block.')
    practice_block = self.make_block()
    for trial_num, trial in enumerate(practice_block):
        self.run_trial(trial, 'practice', trial_num)
    self.display_break()
    self.display_text_screen('Good job! We will now start the real trials.')
```

Then simply pass the hook into run.

```
exp.run(before_first_trial_hook=before_first_trial_hook)
```

Just like that, you have modified the experiment without having to change anything about the underlying implementation!
