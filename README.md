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
