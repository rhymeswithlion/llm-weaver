from .util import get_model_and_tokenizer

def _get_layer_to_weights_map(model):
    from collections import defaultdict
    import re

    layer_to_weights_map = defaultdict(dict)
    for weight in model.weights:
        matches = re.findall(r'/layer_._(\d+)/', weight.name)
        if not matches:
            continue

        layer_number = int(matches[0])
        layer_to_weights_map[layer_number][weight.name.partition(f"/layer_._{layer_number}/")[-1]] = weight

    return {
        layer_number: dict(weights)
        for layer_number, weights in layer_to_weights_map.items()
    }
    
def assign_weights_from_one_layer_to_another(source_model, target_model, source_layer_number, target_layer_number):

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
        assert weight_object.shape == target_layer[weight_name].shape
    
    # Assign weights from one layer to another
    for weight_name, weight_object in source_layer.items():
        target_layer[weight_name].assign(weight_object.numpy())


def weave_models(target_model_template, layer_assignments):
    if isinstance(target_model_template, str):
        target_model, target_tokenizer = get_model_and_tokenizer(target_model_template)
    else:
        target_model = target_model_template

    source_model_names = set(
        layer_assignment["params"]["donor"]
        for layer_assignment in layer_assignments
    )
    source_models = {
        source_model_name: get_model_and_tokenizer(source_model_name)[0]
        for source_model_name in source_model_names
    }

    for layer_assignment in layer_assignments:
        if layer_assignment["type"] == "SingleLayer":
            assign_weights_from_one_layer_to_another(
                source_model=source_models[layer_assignment["params"]["donor"]],
                target_model=target_model,
                source_layer_number=layer_assignment["params"]["hidden_layer_number"],
                target_layer_number=layer_assignment["params"]["hidden_layer_number"]
            )
        else:
            raise NotImplementedError(f"Unknown layer assignment type: {layer_assignment['type']}")

    return target_model


    