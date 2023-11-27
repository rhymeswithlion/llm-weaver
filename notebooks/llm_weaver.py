import tensorflow as tf
from transformers import AutoTokenizer, RobertaConfig
from transformers.models.roberta.modeling_tf_roberta import (
    TFRobertaForSequenceClassification,
)

import model_merging.data as data
import model_merging.evaluation as evaluation
import model_merging.hdf5_util as hdf5_util

def evaluate_model(model, dataset: tf.data.Dataset, metric):
    # Let's check again that all the weights are non-zero
    for name, quantity in get_mean_weights_squared(model).items():
        if quantity <= 1e-10:
            print(f"WARNING: {name} is still zero ({quantity})")
    for model_input, gold_references in dataset:
        model_predictions = model(model_input).logits
        model_predictions = tf.argmax(model_predictions, axis=-1)

        # If all the predictions are the same, print a warning
        if len(tf.unique(model_predictions).y) == 1:
            print(
                f"WARNING: All predictions are the same! Is your model broken? {tf.unique(model_predictions).y}"
            )

        metric.add_batch(predictions=model_predictions, references=gold_references)
    return metric.compute()


def get_score_from_named_model(
    model_id, split, n_examples, max_length=128, batch_size=64
):
    model, tokenizer = get_model_and_tokenizer(model_id)
    task = normalize_glue_task_name(model_id)
    return get_score_from_model(
        model=model,
        tokenizer=tokenizer,
        task=task,
        split=split,
        n_examples=n_examples,
        max_length=max_length,
        batch_size=batch_size,
    )


def get_score_from_model(
    model, tokenizer, task, split, n_examples, max_length=128, batch_size=64
):
    task = normalize_glue_task_name(task)

    ds = data.load_glue_dataset(
        task=task,
        split=split,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    ds = ds.take(n_examples).cache().batch(batch_size)
    # load_metric_for_glue_task doesn't like the "-" in the task name, so we remove it
    metric = evaluation.load_metric_for_glue_task(task.replace("-", ""))
    score = evaluate_model(model, ds, metric)
    return score


# Some tests we did
# tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-RTE")
# get_score_from_model(
#     # model=model,
#     model=get_model("textattack/roberta-base-RTE"),
#     tokenizer=tokenizer,
#     task="rte",
#     split="validation",
#     n_examples=256,
# )
# {'accuracy': 0.7265625}
# This has the same score as taking all the parts of roberta-base-RTE and putting them together using the weave_models function
# calculate_score_from_weaving_config(weaving_configs[0])
# {'accuracy': 0.7265625}

# get_score_from_model(
#     model=get_blank_model(get_model("textattack/roberta-base-RTE").config.to_dict()),
#     tokenizer=tokenizer,
#     task="rte",
#     split="validation",
#     n_examples=256,
# )
# All PyTorch model weights were used when initializing TFRobertaForSequenceClassification.

# All the weights of TFRobertaForSequenceClassification were initialized from the PyTorch model.
# If your task is similar to the task the model of the checkpoint was trained on, you can already use TFRobertaForSequenceClassification for predictions without further training.
# /Users/briancruz/2023-fall-cs-194-294-merging-llms/.venv/lib/python3.8/site-packages/transformers/data/processors/glue.py:520: FutureWarning: This processor will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets library. You can have a look at this example script for pointers: https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
#   warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)
# WARNING: All predictions are the same! Is your model broken? [0]
# WARNING: All predictions are the same! Is your model broken? [0]
# WARNING: All predictions are the same! Is your model broken? [0]


# def calculate_score_from_config()


def calculate_score_from_weaving_config(
    weaving_config, split="validation", n_examples=256
):
    # show md5sum of weaving_config
    import hashlib
    import json

    md5sum = hashlib.md5(
        json.dumps(weaving_config, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    print(f"calculating score for weaving config md5sum: {md5sum}")

    # get model
    model = weave_models(**weaving_config)

    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(weaving_config["tokenizer_model_id"])

    # calculate score
    score = get_score_from_model(
        model=model,
        tokenizer=tokenizer,
        task=weaving_config["glue_task"],
        split=split,
        n_examples=n_examples,
    )
    return score


def get_model(model_str):
    from transformers import TFRobertaForSequenceClassification

    model = TFRobertaForSequenceClassification.from_pretrained(model_str, from_pt=True)
    return model


def get_tokenizer(model_str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_str)
    return tokenizer


def get_model_config(model_str):
    model = get_model(model_str)
    config = model.config.to_dict()
    del config["_name_or_path"]
    return config


def get_blank_model(config):
    blank_model = TFRobertaForSequenceClassification(RobertaConfig(**config))
    blank_model.build()

    # Zero all the weights
    for w in blank_model.weights:
        w.assign(tf.zeros_like(w))

    return blank_model


def dict_overwrite(d1, d2):
    d1 = d1.copy()
    for k in d2:
        d1[k] = d2[k]
    return d1


# The functions we need for model weaving

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification


def get_model_and_tokenizer(identifier):
    tokenizer = AutoTokenizer.from_pretrained(identifier)
    model = TFAutoModelForSequenceClassification.from_pretrained(
        identifier, from_pt=True
    )
    return model, tokenizer


def _get_layer_to_weights_map(model):
    import re
    from collections import defaultdict

    layer_to_weights_map = defaultdict(dict)
    for weight in model.weights:
        matches = re.findall(r"/layer_._(\d+)/", weight.name)
        if not matches:
            continue

        layer_number = int(matches[0])
        layer_to_weights_map[layer_number][
            weight.name.partition(f"/layer_._{layer_number}/")[-1]
        ] = weight

    return {
        layer_number: dict(weights)
        for layer_number, weights in layer_to_weights_map.items()
    }


# def assign_weights_from_one_layer_to_another(
#     source_model, target_model, source_layer_number, target_layer_number
# ):
#     # This part is recalculated often, but it's fast. In the future we could
#     # cache it in a class as a cached property, but we'll leave it here for now.
#     target_model_layer_to_weights_map = _get_layer_to_weights_map(target_model)
#     source_model_layer_to_weights_map = _get_layer_to_weights_map(source_model)

#     # Get the layer objects
#     source_layer = source_model_layer_to_weights_map[source_layer_number]
#     target_layer = target_model_layer_to_weights_map[target_layer_number]

#     # Make sure that all the suffixes match
#     assert set(source_layer.keys()) == set(target_layer.keys())

#     # Make sure that all the shapes match
#     for weight_name, weight_object in source_layer.items():
#         assert weight_object.shape == target_layer[weight_name].shape, (
#             weight_name,
#             weight_object.shape,
#             target_layer[weight_name].shape,
#         )

#     # Assign weights from one layer to another
#     for weight_name, weight_object in source_layer.items():
#         target_layer[weight_name].assign(weight_object.numpy())

def add_weights_from_one_layer_to_another(
    source_model, target_model, source_layer_number, target_layer_number, alpha=1.0, beta=1.0, element_wise_multiplier_filename=None
):
    # This part is recalculated often, but it's fast. In the future we could
    # cache it in a class as a cached property, but we'll leave it here for now.
    target_model_layer_to_weights_map = _get_layer_to_weights_map(target_model)
    source_model_layer_to_weights_map = _get_layer_to_weights_map(source_model)

    # Get the layer objects
    source_layer = source_model_layer_to_weights_map[source_layer_number]
    target_layer = target_model_layer_to_weights_map[target_layer_number]

    # Make sure that all the suffixes match
    assert set(source_layer.keys()) == set(target_layer.keys())

    # Make sure that all the shapes match
    for weight_name, weight_object in source_layer.items():
        assert weight_object.shape == target_layer[weight_name].shape, (
            weight_name,
            weight_object.shape,
            target_layer[weight_name].shape,
        )

    # Assign weights from one layer to another
    for weight_name, weight_object in source_layer.items():
        # Note, weight_name is of the form attention/self/query/kernel:0
        if element_wise_multiplier_filename is not None:
            # Gross hacky way to get the full variable name
            full_variable_name = f"tf_roberta_for_sequence_classification/roberta/encoder/layer_._{source_layer_number}/{weight_name}"
            element_wise_multiplier = get_variable_from_h5_file(element_wise_multiplier_filename, full_variable_name)
            result = alpha * weight_object.numpy() * element_wise_multiplier.numpy() + beta * target_layer[weight_name].numpy()
        else:
            result = alpha * weight_object.numpy() + beta * target_layer[weight_name].numpy()

        target_layer[weight_name].assign(result)

# For each of the variables in the model, return the mean weights squared
def get_mean_weights_squared(model):
    return {
        weight.name: tf.reduce_mean(tf.square(weight)).numpy()
        for weight in model.weights
    }

def get_canonical_variable_name(
    variable_name, base="tf_roberta_for_sequence_classification"
):
    """Returns the canonical variable name for a given tf variable

    Args:
        layer_name (str): The name of the layer

    Examples:
        >>> get_canonical_variable_name("tf_roberta_for_sequence_classification_2/roberta/encoder/layer_._10/output/dense/kernel:0")
        "roberta/encoder/layer_._10/output/dense/kernel"

        >>> get_canonical_variable_name("fisher/tf_roberta_for_sequence_classification/roberta/encoder/layer_._12/output/dense/kernel:0:0")
        "roberta/encoder/layer_._10/output/dense/kernel"
    """
    import re
    # print("==")
    # print(f"before: {variable_name}")
    variable_name = re.sub(r"(^|(^.*/))" + base + r"(_\d+)?/", "", variable_name)
    variable_name = variable_name.partition(":")[0]
    # print(f"after: {variable_name}")
    return variable_name

VARIABLES_FROM_H5_FILE_CACHE = {}

def get_variable_from_h5_file(h5_file, variable_name):
    """Returns a variable from an h5 file"""

    search_name = get_canonical_variable_name(variable_name)

    if h5_file not in VARIABLES_FROM_H5_FILE_CACHE:
        VARIABLES_FROM_H5_FILE_CACHE[h5_file] = hdf5_util.load_variables_from_hdf5(h5_file)
    variables = VARIABLES_FROM_H5_FILE_CACHE[h5_file]
    for variable in variables:
        # print(get_canonical_variable_name(variable.name), search_name)
        if get_canonical_variable_name(variable.name) == search_name:
            return variable
    raise ValueError(f"Could not find variable {variable_name} in h5 file {h5_file}")

def weave_models(
    blank_model_config,
    layer_assignments,
    classification_head=None,
    embeddings=None,
    **kwargs,
):
    # Create a blank model
    target_model = get_blank_model(blank_model_config)

    for name, quantity in get_mean_weights_squared(target_model).items():
        if quantity > 1e-12:
            raise ValueError(f"blank model not zeroed properly. {name} is not zero ({quantity})")

    # We gather all the names of the donor models we need to load
    source_model_names = set(
        layer_assignment["params"]["donor"] for layer_assignment in layer_assignments
        if layer_assignment["type"] == "SingleLayer"
    )
    source_model_names.update(
        donor_configs["donor"] for layer_assignment in layer_assignments
        if layer_assignment["type"] == "IsotropicLinearCombination"
        for donor_configs in layer_assignment["params"]["donors"]
    )
    

    if classification_head is not None:
        source_model_names.add(classification_head["params"]["donor"])
    if embeddings is not None:
        source_model_names.add(embeddings["params"]["donor"])

    # We load all the donor models into a dictionary for easy access
    source_models = {}
    for source_model_name in source_model_names:
        print(f"Loading {source_model_name}")
        source_models[source_model_name] = get_model(source_model_name)

    # Ensure that there is a layer assignment for all the target layers
    assert len(layer_assignments) == blank_model_config["num_hidden_layers"]

    for target_layer_number, layer_assignment in enumerate(layer_assignments):
        if layer_assignment["type"] == "SingleLayer":
            add_weights_from_one_layer_to_another(
                source_model=source_models[layer_assignment["params"]["donor"]],
                target_model=target_model,
                source_layer_number=layer_assignment["params"]["hidden_layer_number"],
                target_layer_number=target_layer_number,
                alpha=1.0, # 100% of the source model
            )
        elif layer_assignment["type"] == "IsotropicLinearCombination":
            for donor_config in layer_assignment["params"]["donors"]:
                add_weights_from_one_layer_to_another(
                    source_model=source_models[donor_config["donor"]],
                    target_model=target_model,
                    source_layer_number=donor_config["hidden_layer_number"],
                    target_layer_number=target_layer_number,
                    alpha=donor_config["weight"],
                )
        elif layer_assignment["type"] == "ElementWiseLinearCombination":
            for donor_config in layer_assignment["params"]["donors"]:
                add_weights_from_one_layer_to_another(
                    source_model=source_models[donor_config["donor"]],
                    target_model=target_model,
                    source_layer_number=donor_config["hidden_layer_number"],
                    target_layer_number=target_layer_number,
                    alpha=donor_config["weight"],
                    element_wise_multiplier_filename=donor_config["element_wise_multiplier_filename"],
                )
        else:
            raise NotImplementedError(
                f"Unknown layer assignment type: {layer_assignment['type']}"
            )

    if classification_head is not None:
        if classification_head["type"] == "SingleClassificationHead":
            # Make sure we have the same number of weights
            assert len(target_model.classifier.weights) == len(
                source_models[classification_head["params"]["donor"]].classifier.weights
            )

            num_weights = len(target_model.classifier.weights)

            assert num_weights == 4  # This is true for roberta-base, but it may change

            for weight_idx in range(num_weights):
                target_weight = target_model.classifier.weights[weight_idx]
                source_weight = source_models[
                    classification_head["params"]["donor"]
                ].classifier.weights[weight_idx]

                assert target_weight.shape == source_weight.shape, (
                    target_weight.name,
                    target_weight.shape,
                    source_weight.name,
                    source_weight.shape,
                )
                assert (
                    target_weight.name.partition("/")[2]
                    == source_weight.name.partition("/")[2]
                )

                target_weight.assign(source_weight.numpy())
            # raise NotImplementedError("TODO: Kirthi")
        else:
            raise NotImplementedError(
                f"Unknown classification head type: {classification_head['type']}"
            )

    if embeddings is not None:
        if embeddings["type"] == "SingleEmbeddings":
            # Make sure we have the same number of weights
            assert len(target_model.roberta.embeddings.weights) == len(
                source_models[embeddings["params"]["donor"]].roberta.embeddings.weights
            )
            num_weights = len(target_model.roberta.embeddings.weights)

            assert num_weights == 5  # This is true for roberta-base, but it may change

            for weight_idx in range(num_weights):
                target_weight = target_model.roberta.embeddings.weights[weight_idx]
                source_weight = source_models[
                    embeddings["params"]["donor"]
                ].roberta.embeddings.weights[weight_idx]

                assert target_weight.shape == source_weight.shape, (
                    target_weight.name,
                    target_weight.shape,
                    source_weight.name,
                    source_weight.shape,
                )
                assert (
                    target_weight.name.partition("/")[2]
                    == source_weight.name.partition("/")[2]
                )

                target_weight.assign(source_weight.numpy())
        else:
            raise NotImplementedError(f"Unknown embeddings type: {embeddings['type']}")

    for name, quantity in get_mean_weights_squared(target_model).items():
        if quantity <= 1e-12:
            print(f"WARNING: {name} is still zero ({quantity})")

    return target_model


def normalize_glue_task_name(task):
    VALID_TASK_NAMES = [
        "cola",
        "mnli",
        # "mnli-mm",
        "mrpc",
        "sst-2",
        "sts-b",
        "qqp",
        "qnli",
        "rte",
        "wnli",
    ]
    for valid_name in VALID_TASK_NAMES:
        if valid_name in task.lower():
            return valid_name
        if valid_name.replace("-", "") in task.lower().replace("-", ""):
            return valid_name
    print(f"WARNING: Could not normalize task name: {task}")
    return task


def test_weaver(original_model_id):
    blank_model_config = get_model_config(original_model_id)

    # fine tuning task is one of:
    # glue_processors = {
    #     "cola": ColaProcessor,
    #     "mnli": MnliProcessor,
    #     "mnli-mm": MnliMismatchedProcessor,
    #     "mrpc": MrpcProcessor,
    #     "sst-2": Sst2Processor,
    #     "sts-b": StsbProcessor,
    #     "qqp": QqpProcessor,
    #     "qnli": QnliProcessor,
    #     "rte": RteProcessor,
    #     "wnli": WnliProcessor,
    # }
    task = normalize_glue_task_name(original_model_id)
    weaved_config = {
        "glue_task": task,
        "tokenizer_model_id": original_model_id,
        # The task (i.e. the classification head output size should match the task at hand)
        "blank_model_config": blank_model_config,
        # Layer assignments
        "layer_assignments": [
            {
                "type": "SingleLayer",
                "params": {
                    # Load donor model
                    "donor": original_model_id,
                    # Pick a layer
                    "hidden_layer_number": i,
                },
            }
            for i in range(blank_model_config["num_hidden_layers"])
        ],
        # The head (i.e. the classification head should match the task at hand)
        # THESE ARE DIFFERENT BETWEEN RTE AND MNLI
        "classification_head": {
            "type": "SingleClassificationHead",
            "params": {
                "donor": original_model_id,
            },
        },
        # The embeddings layer
        # THESE ARE DIFFERENT BETWEEN RTE AND MNLI
        "embeddings": {
            "type": "SingleEmbeddings",
            "params": {
                "donor": original_model_id,
            },
        },
    }

    linear_combo_weaved_config = {
        "glue_task": task,
        "tokenizer_model_id": original_model_id,
        # The task (i.e. the classification head output size should match the task at hand)
        "blank_model_config": blank_model_config,
        # Layer assignments
        "layer_assignments": [
            {
                "type": "IsotropicLinearCombination",
                "params": {
                    "donors": [
                        {
                            "donor": original_model_id,
                            "hidden_layer_number": i,
                            "weight": (0.1*inner_i + 0.1), # 0.1 + 0.2 + 0.3 + 0.4 = 1.0
                        }
                        for inner_i in range(4)
                    ]
                },
            }
            for i in range(blank_model_config["num_hidden_layers"])
        ],
        # The head (i.e. the classification head should match the task at hand)
        # THESE ARE DIFFERENT BETWEEN RTE AND MNLI
        "classification_head": {
            "type": "SingleClassificationHead",
            "params": {
                "donor": original_model_id,
            },
        },
        # The embeddings layer
        # THESE ARE DIFFERENT BETWEEN RTE AND MNLI
        "embeddings": {
            "type": "SingleEmbeddings",
            "params": {
                "donor": original_model_id,
            },
        },
    }

    fisher_weaved_config = {
        "glue_task": task,
        "tokenizer_model_id": original_model_id,
        # The task (i.e. the classification head output size should match the task at hand)
        "blank_model_config": blank_model_config,
        # Layer assignments
        "layer_assignments": [
            {
                "type": "ElementWiseLinearCombination",
                "params": {
                    "donors": [
                        {
                            "donor": original_model_id,
                            "hidden_layer_number": i,
                            "weight": 1.0, 
                            "element_wise_multiplier_filename": None,
                        },
                        {
                            "donor": original_model_id,
                            "hidden_layer_number": i,
                            "weight": 1.0,
                            "element_wise_multiplier_filename": f"../data/fisher_info/{original_model_id.replace('/', '_')}-fisher-info.h5",
                        },
                        {
                            "donor": original_model_id,
                            "hidden_layer_number": i,
                            "weight": -1.0,
                            "element_wise_multiplier_filename": f"../data/fisher_info/{original_model_id.replace('/', '_')}-fisher-info.h5",
                        },
                    ]
                },
            }
            for i in range(blank_model_config["num_hidden_layers"])
        ],
        # The head (i.e. the classification head should match the task at hand)
        # THESE ARE DIFFERENT BETWEEN RTE AND MNLI
        "classification_head": {
            "type": "SingleClassificationHead",
            "params": {
                "donor": original_model_id,
            },
        },
        # The embeddings layer
        # THESE ARE DIFFERENT BETWEEN RTE AND MNLI
        "embeddings": {
            "type": "SingleEmbeddings",
            "params": {
                "donor": original_model_id,
            },
        },
    }
    fisher_weaved_score = calculate_score_from_weaving_config(
        fisher_weaved_config,
        n_examples=100,
        split="validation",
    )
    
    linear_combo_weaved_score = calculate_score_from_weaving_config(
        linear_combo_weaved_config,
        n_examples=100,
        split="validation",
    )

    weaved_score = calculate_score_from_weaving_config(
        weaved_config,
        n_examples=100,
        split="validation",
    )

    original_score = get_score_from_model(
        model=get_model(original_model_id),
        tokenizer=get_tokenizer(original_model_id),
        task=task,
        split="validation",
        n_examples=100,
    )

    print(f"Original score ({original_model_id}):", original_score)
    print(f"Weaved score ({original_model_id}):", weaved_score)
    print(f"Linear combo weaved score ({original_model_id}):", linear_combo_weaved_score)
    print(f"Fisher weaved score ({original_model_id}):", fisher_weaved_score)

    scores = [original_score, weaved_score, linear_combo_weaved_score, fisher_weaved_score]
    configs = [None, weaved_config, linear_combo_weaved_config, fisher_weaved_config]

    if all("accuracy" in score for score in scores):
        # assert all accuracy scores are the same
        assert all(
            abs(score["accuracy"] - original_score["accuracy"]) < 1e-3
            for score in scores
        )

    return scores, configs
