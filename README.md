# Optimal Transport Posterior Alignment for Cross-lingual Semantic Parsing
### Tom Sherborne, Tom Hosking, Mirella Lapata
### @tomsherborne @tomhosking 

To appear in TACL 2023 and presented at EMNLP 2023.
Read the ArXiv version [here](https://arxiv.org/abs/2307.04096)

This version of the `minotaur` codebase is for @evgeniiaraz 

-------

Hi Genie!

This is a run down of the `minotaur` code to help you get started. This project uses [AllenNLP](https://github.com/allenai/allennlp) (RIP) and the full environment is in `requirements.txt`.

You specify an experiment as a `json` config file and then run this using:

```bash
EXP_CONFIG=/path/to/jsonconfig
SERIAL_DIR=/place/to/output/experiment
allennlp train "${EXP_CONFIG}" \
        --serialization-dir "${SERIAL_DIR}" \
        --include-package codebase \
        --file-friendly-logging \
        --force
```

I've included two example JSON files as examples. Each experiment needs these keys:

    1. "dataset_reader" - How to read text file inputs to `Instance` objects for training
    2.  "data_loader" - How data is loaded from the dataset readers and passed to the trainer. There will also be a "validation_data_loader" if the validation data is loaded differently to the Train data. 
    3. "model" - specification of the model to train
    4. "train_data_path" - path to the Training data
    5. "validation_data_path" - path to the validation data
    6. "trainer" - the Trainer to use

Because we have two data streams for training ('inner' and 'outer'), the project looks more complicated than a simpler training setup. You don't have to use this code if you just want to take the variational alignment code independently. The code is setup
to use cached pre-trained models under `big/mbart-large-50-many-to-many-mmt` which you can download from HF separately. The `atis_exp_template.sh` describes a SLURM job run through of an experiment.

If you want to change anything in this setup, you can change the "type" in the JSON config to registered name in the 
decorator above the class name. Each section in the JSON corresponds to the __init__ of each class and any subclass objects spawn a child JSON bracket (e.g. the "decoder" subsection of the "model").

The most important information for Minotaur is:

    - `model/bottleneck.py`: defines the variational reparameterization between Encoder and Decoder. This makes the Seq2Seq model like a VAE/WAE and provides the ELBO loss. 

    - `model/seq2seq_bottlneck.py`: defines the Seq2Seq model using a `Bottleneck` between Encoder and Decoder. 

    - `trainer/divergence_kernel.py`: defines the Alignment signal between encoded states. Looks a lot like `Bottleneck` but only calculates distances between given Tensors of states. 
    
    - `trainer/episodic_trainer2.py`: This is the training loop tying everything together. Read this if you want to see how I built the model. See `trainer/episodic_trainer4_nonpar.py` for how the same loop runs when the encoded states are not parallel.

Everywhere should have annotations and detail about Tensor sizes to guide you. Let me know if you have questions/need a hand! 
