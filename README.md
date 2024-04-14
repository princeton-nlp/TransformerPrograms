# Learning Transformer Programs

This repository contains the code for our reproduction of the [Learning Transformer Programs](https://arxiv.org/abs/2306.01128).
The code can be used to train a modified Transformer to solve RASP tasks and convert it into a human-readable Python program.

<img src="figures/methodology.png" width="100%" height="auto"/>

## Quick links
* [Setup](#Setup)
* [Learning Programs](#Learning-programs)
  * [Training](#Training)
  * [Converting to code](#Converting-to-code)
* [Results](#Results)
* [Citation](#Citation)

## Setup

Install [PyTorch](https://pytorch.org/get-started/locally/) and install the remaining requirements: `pip install -r requirements.txt`.
This code was tested using Python 3.8 and PyTorch version 1.13.1.

## Learning Programs

### Training

The code to learn a Transformer Program can be found in [src/run.py](src/run.py).
For example, the following command will train a Transformer Program for the `sort` task, using two layers, four categorical attention heads per layer, and one-hot input embedding:
```bash
python ../src/run.py \
   --dataset "sort" \
   --dataset_size 20000 \
   --train_min_length 1 \
   --train_max_length 8 \
   --test_min_length 1 \
   --test_max_length 8 \
   --d_var 8 \
   --n_heads_cat 4 \
   --n_heads_num 4 \
   --n_cat_mlps 2 \
   --n_num_mlps 2 \
   --n_layers 3 \
   --one_hot_embed \
   --count_only \
   --save \
   --save_code \
   --output_dir "output/sort;
```
Please see [src/run.py](src/run.py) for all of the possible arguments.
The training data will be generated before training for the RASP tasks; See [src/utils/data_utils.py](src/utils/data_utils.py) for the supported datasets.
The [scripts](scripts/) directory contains scripts for training Transformer Programs with the experiment settings used in the paper.

### Converting to code

Run the training script with the `--save_code` flag to convert the model to a Python program at the end of training.
To convert a model that has already been trained, use `src/decompile.py`.
For example,
```bash
python src/decompile.py --path output/sort/ --output_dir programs/sort/
```
`output/sort/` should be the output directory of a training run.

## Results

Below are our reproduction results of the paper.

# Contributions
<table>
    <tr>
        <th>Member</th>
        <th>Contribution</th>
    </tr>
    <tr>
        <td>Nina Oosterlaar</td>
        <td>Length generalization and grid search implementation</td>
    </tr>
    <tr>
        <td>Sebastiaan Beekman</td>
        <td>Refactoring, gridsearch implementation, data collection, and debugging</td>
    </tr>
    <tr>
        <td>Joyce Sung</td>
        <td>Length generalization and addition implementation</td>
    </tr>
    <tr>
        <td>Zoya van Meel</td>
        <td>Addition implementation, data visualisation, and poster design</td>
    </tr>
    <tr>
        <td colspan=2>Everyone contributed equally to the final blog</td>
    </tr>
</table>


# Citation

```bibtex
@article{friedman2023learning,
    title={Learning {T}ransformer {P}rograms},
    author={Friedman, Dan and Wettig, Alexander and Chen, Danqi},
    journal={arXiv preprint},
    year={2023}
}
```
