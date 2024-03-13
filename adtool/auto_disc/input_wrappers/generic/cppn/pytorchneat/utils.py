import torch


def create_image_cppn_input(output_size, is_distance_to_center=True, is_bias=True):
    r = 1.0  # std(coord_range) == 1.0
    coord_ranges = []
    for output_size_dim in output_size:
        coord_ranges.append(torch.linspace(-r, r, output_size_dim))
    meshgrids = torch.meshgrid(*coord_ranges)
    cppn_input = torch.stack(list(meshgrids), -1)
    if is_distance_to_center:
        d = torch.linalg.norm(cppn_input, dim=-1).unsqueeze(-1)
        cppn_input = torch.cat([cppn_input, d], -1)
    if is_bias:
        cppn_input = torch.cat([cppn_input, torch.ones(output_size).unsqueeze(-1)], -1)

    return cppn_input
