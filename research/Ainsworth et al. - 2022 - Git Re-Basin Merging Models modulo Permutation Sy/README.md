# Merging Models with Fisher-Weighted Averaging

```bibtex
@misc{matena_merging_2022,
	title = {Merging {Models} with {Fisher}-{Weighted} {Averaging}},
	url = {http://arxiv.org/abs/2111.09832},
	abstract = {Averaging the parameters of models that have the same architecture and initialization can provide a means of combining their respective capabilities. In this paper, we take the perspective that this "merging" operation can be seen as choosing parameters that approximately maximize the joint likelihood of the posteriors of the models' parameters. Computing a simple average of the models' parameters therefore corresponds to making an isotropic Gaussian approximation to their posteriors. We develop an alternative merging procedure based on the Laplace approximation where we approximate each model's posterior as a Gaussian distribution whose precision matrix corresponds to its Fisher information. We first show that our "Fisher merging" technique provides a performance boost in settings where simple parameter averaging is currently used -- specifically, robust fine-tuning and model ensembling. Then, we compare merging to standard gradient-based transfer learning and demonstrate that merging enables a fundamentally different method for transferring capabilities across models. Specifically, we show that Fisher merging is competitive with gradient-based transfer learning approaches (while being significantly cheaper) in intermediate-task training and domain-adaptive pre-training. We also show that our merging procedure makes it possible to combine models in previously unexplored ways. We release our code to facilitate future research into methods for merging models.},
	urldate = {2023-10-31},
	publisher = {arXiv},
	author = {Matena, Michael and Raffel, Colin},
	month = aug,
	year = {2022},
	note = {arXiv:2111.09832 [cs]},
	keywords = {Computer Science - Machine Learning},
	file = {arXiv.org Snapshot:/Users/testaccount/Zotero/storage/BYL4NYFJ/2111.html:text/html;Matena and Raffel - 2022 - Merging Models with Fisher-Weighted Averaging.pdf:/Users/testaccount/Zotero/storage/2PDENJMJ/Matena and Raffel - 2022 - Merging Models with Fisher-Weighted Averaging.pdf:application/pdf},
}
```

Here is the link: https://github.com/mmatena/model_merging
