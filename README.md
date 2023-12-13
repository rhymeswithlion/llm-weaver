# Exploration of Transformer Model Layer Interaction and Optimal Merging Strategies

This is our repository for our project for CS194/294 (Exploration of Transformer Model Layer Interaction and Optimal Merging Strategies)

```
.
├── README.md
├── data
│   ├── fisher_info    <----- Fisher information matrices go here
├── environment.yml
├── model_merging      <----- This is a clone of the repo https://github.com/mmatena/model_merging
│   ├── README.html           We used this repository to calculate fisher information matricies
│   ├── README.md             and also to run evaluations on Glue tasks
│   ├── model_merging
│   ├── pyproject.toml
│   └── scripts
├── notebooks          <---------------------- Our notebooks!
│   ├── alpha-beta-experiment.ipynb       <--- Alpha beta experiment. The most recent version of this was lost.
│   ├── cache
│   ├── extra_notebooks
│   ├── llm_weaver.py
│   ├── markov-transitions-heatmap-roberta-base-mnli.ipynb
│   ├── markov-transitions-heatmap-roberta-base-rte.ipynb
│   ├── markov-transitions-heatmap-roberta-base-sst2.ipynb
│   ├── markov-transitions-heatmap-roberta-large-rte.ipynb
│   ├── model-weaving-experiement-part-a.ipynb
│   └── model-weaving-experiment-part-b.ipynb
├── scripts_for_generating_fisher_weights
```
