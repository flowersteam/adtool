import torch


def create_image_cppn_input(output_size, is_distance_to_center=True, is_bias=True):
    r = 1.0  
    coord_ranges = []
    for output_size_dim in output_size[:-1]:
        coord_ranges.append(torch.linspace(-r, r, output_size_dim))
    meshgrids = torch.meshgrid(*coord_ranges)
    cppn_input = torch.stack(list(meshgrids), -1)
    if is_distance_to_center:
        d = torch.linalg.norm(cppn_input, dim=-1).unsqueeze(-1)
        
        cppn_input = torch.cat([cppn_input, d], -1)
    if is_bias:
        cppn_input = torch.cat([cppn_input, torch.ones(output_size[:-1]).unsqueeze(-1)], -1)


    cppn_input = cppn_input.unsqueeze(-2).expand(-1, -1, output_size[-1], -1)
        
    
        
         
    return cppn_input
