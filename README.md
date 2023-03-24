# Two-stage-fine-tuning

## Install dependancies
```
pip install torch
pip install transformers
```
The codes were tested with transformes version 4.13.0 and torch version 1.8.1 with a compatible cuda.
## Prepare data
Download and unzip the SST-2 data from GLUE
```
wget https://dl.fbaipublicfiles.com/glue/data/SST-2.zip
unzip SST-2.zip
```
Down sample data
```
python down_sample.py -i SST-2 -o down_sampled
```

## Run the experiments
### Run Two-stage Fine-tuning with reweighting
```
python two_stage_reweight.py --data-path down_sampled/0.2 --output-path results/Two-stage-reweight --epoch-stage-1 1 --epoch-stage-2 1
```

### Run Two-stage Fine-tuning with chatGPT augmented data
First download the augmented data from the link below and put it in the down_sampled/0.2 folder
https://drive.google.com/file/d/1PyTdS9Ev_OhsU2WQSQWRxBw8TV8Z27tB/view?usp=sharing

Then, run the the experiment
```
python two_stage_aug.py --data-path down_sampled/0.2 --output-path results/Two-stage-chatGPT --epoch-stage-1 1 --epoch-stage-2 1
```

### Run Vanilla Fine-tuning (full)
```
python two_stage_reweight.py --data-path down_sampled/0.2 --output-path results/Vanilla --epoch-stage-1 0 --epoch-stage-2 1
```

The results of each experiment will be saved in CSV in the corresponding folders.

### Cite as:
```
@misc{https://doi.org/10.48550/arxiv.2207.10858,
  doi = {10.48550/ARXIV.2207.10858},
  url = {https://arxiv.org/abs/2207.10858},
  author = {ValizadehAslani, Taha and Shi, Yiwen and Wang, Jing and Ren, Ping and Zhang, Yi and Hu, Meng and Zhao, Liang and Liang, Hualou},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Two-Stage Fine-Tuning: A Novel Strategy for Learning Class-Imbalanced Data},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}}
```
