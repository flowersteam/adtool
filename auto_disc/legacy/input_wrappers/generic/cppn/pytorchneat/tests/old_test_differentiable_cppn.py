import math
import os
import time
from unittest import TestCase

import matplotlib.pyplot as plt
import neat
import torch
import torchvision
from pytorchneat import activations, aggregations, rnn, selfconnectiongenome
from torchvision import transforms
from utils import create_image_cppn_input

use_Minkowski_inputs = False
if use_Minkowski_inputs:
    import MinkowskiEngine as ME
    import numpy as np
    import open3d as o3d
    from MinkowskiEngine.MinkowskiFunctional import _wrap_tensor


def delphineat_gauss_activation(z):
    """Gauss activation as defined by SharpNEAT, which is also as in DelphiNEAT."""
    return 2 * math.exp(-1 * (z * 2.5) ** 2) - 1


def delphineat_sigmoid_activation(z):
    """Sigmoidal activation function as defined in DelphiNEAT"""
    return 2.0 * (1.0 / (1.0 + math.exp(-z * 5))) - 1


class TestRecurrentNetwork(TestCase):
    def test_pytorchneat_differentiability(self):
        config_path = os.path.join(os.path.dirname(__file__), "test_neat.cfg")
        neat_config = neat.Config(
            selfconnectiongenome.SelfConnectionGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )
        neat_config.genome_config.add_activation(
            "delphineat_gauss", delphineat_gauss_activation
        )
        neat_config.genome_config.add_activation(
            "delphineat_sigmoid", delphineat_sigmoid_activation
        )

        # create the cppn input (image_height, image_width,num_inputs)
        img_height = 56
        img_width = 56
        cppn_input = create_image_cppn_input((img_height, img_width))
        if use_Minkowski_inputs:
            coords = []
            feats = []
            for i in range(img_height):
                for j in range(img_width):
                    if torch.rand(()) < 0.7:
                        coords.append(torch.tensor([i, j], dtype=torch.float64))
                        feats.append(torch.tensor([i, j, cppn_input[i, j, 2], 1]))
            cppn_input = ME.SparseTensor(torch.stack(feats), torch.stack(coords))
        mnist_dataset = torchvision.datasets.MNIST(
            root="/home/mayalen/data/pytorch_datasets/mnist/",
            download=False,
            train=True,
        )
        upscale_target_tansform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((img_height, img_width)),
                transforms.ToTensor(),
            ]
        )

        if use_Minkowski_inputs:
            vis = o3d.visualization.Visualizer()

        for mnist_target_idx in range(1, 21):
            if not os.path.exists("test_differentiability_outputs"):
                os.makedirs("test_differentiability_outputs")
            if not os.path.exists("test_differentiability_outputs/mnist_targets"):
                os.makedirs("test_differentiability_outputs/mnist_targets")
            cur_output_dir = f"test_differentiability_outputs/mnist_targets/target_{mnist_target_idx}"
            if not os.path.exists(cur_output_dir):
                os.makedirs(cur_output_dir)
            target_img = (
                upscale_target_tansform(mnist_dataset.data[mnist_target_idx])
                .squeeze()
                .float()
            )
            target_img = target_img / target_img.max()
            plt.figure()
            plt.imshow(target_img, cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(cur_output_dir, "target.png"))
            plt.close()
            coords, feats = [], []

            if use_Minkowski_inputs:
                for i, row in enumerate(target_img):
                    for j, val in enumerate(row):
                        if val != 0:
                            coords.append([i, j])
                            feats.append([val])
                target_img = ME.SparseTensor(
                    features=torch.DoubleTensor(feats),
                    coordinates=torch.IntTensor(coords),
                    coordinate_manager=cppn_input.coordinate_manager,
                )

            def eval_genomes(genomes, neat_config):
                genomes_train_losses = []
                genomes_train_images = []

                for genome_idx, genome in genomes:
                    t0 = time.time()
                    cppn_net = rnn.RecurrentNetwork.create(genome, neat_config)
                    opt = torch.optim.Adam(cppn_net.parameters(), 1e-2)
                    train_losses = []
                    train_images = []
                    for train_step in range(45):
                        cppn_net_output = cppn_net.activate(cppn_input, 2)
                        cppn_net_output
                        if isinstance(cppn_net_output, torch.Tensor):
                            cppn_net_output = (1.0 - cppn_net_output.abs()).view(
                                img_height, img_width
                            )
                        elif use_Minkowski_inputs:
                            # .view(img_height, img_width)
                            cppn_net_output = _wrap_tensor(
                                cppn_net_output, 1.0 - cppn_net_output.F.abs()
                            )
                        loss = (target_img - cppn_net_output).F.pow(2).sum()
                        opt.zero_grad()
                        loss.backward()
                        opt.step()

                        train_losses.append(loss.item())
                        if isinstance(cppn_net_output, torch.Tensor):
                            train_images.append(
                                cppn_net_output.cpu().detach().unsqueeze(0)
                            )
                        elif use_Minkowski_inputs:
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(
                                torch.cat(
                                    [
                                        cppn_net_output.C,
                                        torch.zeros([cppn_net_output.C.shape[0], 1]),
                                    ],
                                    -1,
                                )
                                .cpu()
                                .detach()
                            )
                            pcd.colors = o3d.utility.Vector3dVector(
                                cppn_net_output.F.repeat(1, 3).cpu().detach()
                            )
                            # o3d.visualization.draw_geometries([pcd])
                            vis.create_window(
                                "pcl", img_width, img_height, 50, 50, True
                            )
                            vis.add_geometry(pcd)
                            # out_depth = vis.capture_depth_float_buffer(True)
                            out_image = vis.capture_screen_float_buffer(True)
                            train_images.append(
                                torch.from_numpy(np.asarray(out_image)).transpose(0, -1)
                            )
                    t1 = time.time()
                    print(f"Training genome {genome_idx} took {t1-t0} secs")

                    genome.fitness = -train_losses[-1]
                    plt.figure()
                    plt.subplot(211)
                    plt.plot(train_losses)
                    plt.subplot(212)
                    plt.imshow(
                        torchvision.utils.make_grid(train_images, 15).permute(1, 2, 0)
                    )
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(cur_output_dir, f"individual_{genome.key}.png")
                    )
                    plt.close()

                    # rewrite trained values in genome:
                    cppn_net.update_genome(genome)

                genomes_train_losses.append(train_losses)
                genomes_train_images.append(train_images)

                return genomes_train_losses, genomes_train_images

            pop = neat.Population(neat_config)
            stats = neat.StatisticsReporter()
            pop.add_reporter(stats)
            reporter = neat.StdOutReporter(True)
            pop.add_reporter(reporter)

            n_generations = 10
            pop.run(eval_genomes, n_generations)

            if use_Minkowski_inputs:
                vis.close()

        return
