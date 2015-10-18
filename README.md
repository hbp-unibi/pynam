PyNAM
======

PyNAM is a spiking neural network implementation of the Willshaw associative memory (BiNAM), implemented in Python with PyNNLess/PyNN as a backend.

It serves as a potential benchmark for neuromorphic hardware systems and as a test bed for experiments with the Willshaw associative memory.

## Usage

To run the complete analysis pipeline, run the program in one of the following
forms:

    ./run.py <SIMULATOR> [<EXPERIMENT>]
    ./run.py <SIMULATOR> --process <EXPERIMENT>

Where `<SIMULATOR>` is the simulator that should be used for execution
and `<EXPERIMENT>` is a JSON file describing the experiment that should be executed.
If `<EXPERIMENT>` is not given, the program will try to execute "experiment.json".

In order to just generate the network descriptions for a specific experiment,
use the following format:

    ./run.py <SIMULATOR> --create <EXPERIMENT>

Note that the `<SIMULATOR>` is needed in order to partition the experiment
according to the hardware resources. This command will generate a set of
".in.gz" files that can be passed to the execution stage:

    ./run.py <SIMULATOR> --exec <IN_1> ... <IN_N>

This command will execute the network simulations for the given network
descriptors -- if more than one network descriptor is given, a new process
will be spawned for each execution. Generates a series of ".out.gz" files
in the same directory as the input files.

    ./run.py <TARGET> --analyse <OUT_1> ... <OUT_N>

Analyses the given output files, generates a HDF5/Matlab file as `<TARGET>`
containing the processed results.

## Simulators

Possible simulators are:

* `spikey`
* `nest`
* `nmmc1`
* `nmpm1`
* `ess`

## Experiment description documentation

The experiment description is read from a JSON-like file format -- in contrast to standard JSON, we allow JavaScript-comments inside the file. The available parameters are as follows:

```javascript
{
    /*
     * The data parameters specify the data that is stored in the Willshaw
     * associative memory and implicitly the network size.
     */
    "data": {
        "n_bits_in": 16, // Number of input bits
        "n_bits_out": 16, // Corresponds to the number of neurons
        "n_ones_in": 3, // Number of ones in the input samples
        "n_ones_out": 3 // Number of ones in the output samples
        //"n_samples": 100 // Number of samples (set automatically if not given)
    },

    /**
     * Network topology and neuron parameters
     */
    "topology": {
        "multiplicity": 1, // Size of a neuron population
        // PyNN neuron parameters (depending on the current model)
        "params": {
            "cm": 0.2,
            "e_rev_E": -40,
            "e_rev_I": -60,
            "v_rest": -50,
            "v_reset": -70,
            "v_thresh": -47
        },
        "param_noise": {
            // Standard deviation of each neuron parameter
        },
        // Neuron model, either IF_cond_exp or EIF_cond_exp_isfa_ista
        "neuron_type": "IF_cond_exp",
        "w": 0.011,  // Synapse weight
        "sigma_w": 0.0 // Standard deviation of the synapse weight
    },

    /**
     * Input data specification
     */
    "input": {
        "burst_size": 1, // Number of spikes in an input burst
        "time_window": 100.0, // Time between samples
        "isi": 1.0, // Time between spikes in a burst
        "sigma_t": 0.0, // Standard deviation of the spike time noise
        "sigma_t_offs": 0.0, // Standard deviation of the burst offset
        "p0": 0.0, // Probability of input spikes missing
        "p1": 0.0 // Probability of a (partial) false-positive input burst
    },

    /**
     * Output data specification
     */
    "output": {
        "burst_size": 1 // Number of spikes representing a "1"
    },

    /**
     * List of experiments to be conducted
     */
    "experiments": [
        {
            "name": "sweep_w", // Name of the experiment
            "sweeps": { // Map of dimensions that should be swept over
                "topology.w": {"min": 0.0, "max": 0.01, "count": 10}
            },
            "repeat": 10 // Number of times the experiment should be repeated
        }
    ]
}
```

## Authors

This project has been initiated by Andreas Stöckel in 2015 as part of his Masters Thesis
at Bielefeld University in the [Cognitronics and Sensor Systems Group](http://www.ks.cit-ec.uni-bielefeld.de/) which is
part of the [Human Brain Project, SP 9](https://www.humanbrainproject.eu/neuromorphic-computing-platform).

## License

This project and all its files are licensed under the
[GPL version 3](http://www.gnu.org/licenses/gpl.txt) unless explicitly stated
differently.



