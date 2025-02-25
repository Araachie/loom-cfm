<h1 align="center">
  <br>
	Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling
  <br>
</h1>
  <p align="center">
    <a href="https://araachie.github.io">Aram Davtyan</a> •
    <a href="https://scholar.google.com/citations?user=bhAxvCIAAAAJ&hl=en">Leello Tadesse Dadi</a> •
    <a href="https://people.epfl.ch/volkan.cevher">Volkan Cevher</a> •
    <a href="https://www.cvg.unibe.ch/people/favaro">Paolo Favaro</a>
  </p>
<h4 align="center">Official repository of the paper</h4>

<h4 align="center">at ICLR 2025</h4>

<h4 align="center"><a href="https://araachie.github.io/loom-cfm/">Website</a> • <a href="https://openreview.net/forum?id=rsGPrJDIhh">Paper</a>

#
> **Abstract:** *Conditional Flow Matching (CFM), a simulation-free method for training continuous normalizing
> flows, provides an efficient alternative to diffusion models for key tasks like image and video generation.
> The performance of CFM in solving these tasks depends on the way data is coupled with noise. A recent approach
> uses minibatch optimal transport (OT) to reassign noise-data pairs in each training step to streamline sampling
> trajectories and thus accelerate inference. However, its optimization is restricted to individual minibatches,
> limiting its effectiveness on large datasets. To address this shortcoming, we introduce LOOM-CFM (Looking Out Of Minibatch-CFM),
> a novel method to extend the scope of minibatch OT by preserving and optimizing these assignments across minibatches over training time.
> Our approach demonstrates consistent improvements in the sampling speed-quality trade-off across multiple datasets.
> LOOM-CFM also enhances distillation initialization and supports high-resolution synthesis in latent space training.*

## Citation

```
@inproceedings{
  davtyan2025faster,
  title={Faster Inference of Flow-Based Generative Models via Improved Data-Noise Coupling},
  author={Aram Davtyan and Leello Tadesse Dadi and Volkan Cevher and Paolo Favaro},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=rsGPrJDIhh}
}
```

## Prerequisites

For convenience, we provide an `environment.yml` file that can be used to install the required packages 
to a `conda` environment with the following command 

```conda env create -f environment.yml```

The code was tested with cuda=12.1 and python=3.9.

## Pretrained models

We share the weights of the models pretrained on the datasets considered in the paper.

<table style="margin:auto">
    <thead>
        <tr>
          <th>Model name</th>
          <th>Model weights</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>CIFAR10</td>
            <td><a href="https://huggingface.co/cvg-unibe/loom-cfm_cifar10/blob/main/model.pth">download</a></td>
        </tr>
        <tr>
            <td>CIFAR10_reflow_1x</td>
            <td><a href="https://huggingface.co/cvg-unibe/loom-cfm_cifar10_reflow/blob/main/model.pth">download</a></td>
        </tr>
        <tr>
            <td>ImageNet32</td>
            <td><a href="https://huggingface.co/cvg-unibe/loom-cfm_imagenet32/blob/main/model.pth">download</a></td>
        </tr>
        <tr>
            <td>ImageNet64</td>
            <td><a href="https://huggingface.co/cvg-unibe/loom-cfm_imagenet64/blob/main/model.pth">download</a></td>
        </tr>
        <tr>
            <td>FFHQ256</td>
            <td><a href="https://huggingface.co/cvg-unibe/loom-cfm_ffhq256/blob/main/model.pth">download</a></td>
        </tr>
    </tbody>
</table>

## Running pretrained models

To sample from pre-trained models, use the `sample.py` script from this repository.

```
usage: sample.py [-h] --config CONFIG --ckpt CKPT --output OUTPUT --num_images NUM_IMAGES [--nrows NROWS] --method METHOD [--batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the config file.
  --ckpt CKPT           Path to the model checkpoint.
  --output OUTPUT       Path to the output directory to write images to.
  --num_images NUM_IMAGES
                        Number of images to generate.
  --nrows NROWS         nrows in make_grid.
  --method METHOD       Generation method of format solver-steps.
  --batch_size BATCH_SIZE
                        Batch size.
```

## Training your own model

To train your own model you first need to start by preparing the data.

### Datasets

A custom dataset can be organized either as a `torchvision.datasets.ImageFolder` dataset (see `configs/imagenet64_ot_cached.yaml` as an example) or stored in `HDF5` files (see `configs/ffhq256_latent_ot_cached.yaml` for an example).

The dataset folder should be organized as follows:

```
data_root/
|---train/  # contains image folders or h5 files
|---val/  # contains image folders or h5 files
```

### Training

For the training, use the `train.py` script from this repository. Usage example:

```
python train.py --config <path_to_config> --run-name <run_name> --wandb
```

The output of `python train.py --help` is as follows:

```
usage: train.py [-h] --run-name RUN_NAME --config CONFIG [--num-gpus NUM_GPUS] [--resume-step RESUME_STEP] [--random-seed RANDOM_SEED] [--wandb]

optional arguments:
  -h, --help            show this help message and exit
  --run-name RUN_NAME   Name of the current run.
  --config CONFIG       Path to the config file.
  --num-gpus NUM_GPUS   Number of gpus to use for training.By default uses all available gpus.
  --resume-step RESUME_STEP
                        Step to resume the training from.
  --random-seed RANDOM_SEED
                        Random seed.
  --wandb               If defined, use wandb for logging.
```

Use the configs in the `configs` folder as guidance.

### Training in latent space

To train in the latent space of a pre-trained autoencoder, provide the details about the autoencoder in the model config, i.e. `config[model][ae_config]` and  `config[model][ae_checkpoint]`. At the moments the supported autoencoders are f8 and f16 VQ and KL autoencoders from [LDM](https://github.com/CompVis/stable-diffusion/tree/main). For more details, check `model/vqgan/taming.autoencoder.py` and `configs/ffhq256_latent_ot_cached.yaml`.

### Reflow

To run iterations of Reflow algorithm, first, one needs to create the dataset of source-target pairs sampled from the model. For this, use the `generate_pairs.py` script.

```
usage: generate_pairs.py [-h] --config CONFIG --ckpt CKPT --output OUTPUT --num_images NUM_IMAGES --method METHOD [--batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the config file.
  --ckpt CKPT           Path to the model checkpoint.
  --output OUTPUT       Path to the output directory to write images to.
  --num_images NUM_IMAGES
                        Number of images to generate.
  --method METHOD       Generation method of format solver-steps.
  --batch_size BATCH_SIZE
                        Batch size.
```

This will create a HDF5 dataset of source-target pairs. To use it for training, you need to change the original config by prepending `"reflow_"` to `config[data][name]` and specifying `config[data][eval_data_root]` (see `configs/cifar10_ot_cached_reflow.yaml` for guidance).

### Evaluation

To evaluate a trained model, use the `evaluate.py` script from this repository.

```
usage: evaluate.py [-h] --config CONFIG --ckpt CKPT

optional arguments:
  -h, --help       show this help message and exit
  --config CONFIG  Path to the config file.
  --ckpt CKPT      Path to the model checkpoint.
```
