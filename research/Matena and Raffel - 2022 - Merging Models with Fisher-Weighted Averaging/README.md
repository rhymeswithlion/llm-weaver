# Git Re-Basin: Merging Models modulo Permutation Symmetries

```bibtex
@misc{ainsworth_git_2022,
	title = {Git {Re}-{Basin}: {Merging} {Models} modulo {Permutation} {Symmetries}},
	shorttitle = {Git {Re}-{Basin}},
	url = {https://arxiv.org/abs/2209.04836v6},
	abstract = {The success of deep learning is due in large part to our ability to solve certain massive non-convex optimization problems with relative ease. Though non-convex optimization is NP-hard, simple algorithms -- often variants of stochastic gradient descent -- exhibit surprising effectiveness in fitting large neural networks in practice. We argue that neural network loss landscapes often contain (nearly) a single basin after accounting for all possible permutation symmetries of hidden units a la Entezari et al. 2021. We introduce three algorithms to permute the units of one model to bring them into alignment with a reference model in order to merge the two models in weight space. This transformation produces a functionally equivalent set of weights that lie in an approximately convex basin near the reference model. Experimentally, we demonstrate the single basin phenomenon across a variety of model architectures and datasets, including the first (to our knowledge) demonstration of zero-barrier linear mode connectivity between independently trained ResNet models on CIFAR-10. Additionally, we identify intriguing phenomena relating model width and training time to mode connectivity. Finally, we discuss shortcomings of the linear mode connectivity hypothesis, including a counterexample to the single basin theory.},
	language = {en},
	urldate = {2023-10-18},
	journal = {arXiv.org},
	author = {Ainsworth, Samuel K. and Hayase, Jonathan and Srinivasa, Siddhartha},
	month = sep,
	year = {2022},
	file = {Ainsworth et al. - 2022 - Git Re-Basin Merging Models modulo Permutation Sy.pdf:/Users/testaccount/Zotero/storage/B9LWTJUC/Ainsworth et al. - 2022 - Git Re-Basin Merging Models modulo Permutation Sy.pdf:application/pdf},
}
```

> Our code is open sourced at https://github.com/samuela/git-re-basin. Our experimental logs and downloadable model checkpoints are fully open source at https://wandb.ai/skainswo/git-re-basin.
