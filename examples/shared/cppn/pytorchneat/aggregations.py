import torch


def sum_aggregation(inputs):
    return torch.sum(inputs, -1, dtype=inputs.dtype)


def product_aggregation(inputs):
    return torch.prod(inputs, -1, dtype=inputs.dtype)


str_to_aggregation = {
    "sum": sum_aggregation,
    "product": product_aggregation,
}
