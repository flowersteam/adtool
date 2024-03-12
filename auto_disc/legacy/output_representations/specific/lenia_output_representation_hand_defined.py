import typing

import torch
from addict import Dict
from auto_disc.legacy.output_representations import BaseOutputRepresentation
from auto_disc.legacy.utils.config_parameters import (
    DecimalConfigParameter,
    IntegerConfigParameter,
    StringConfigParameter,
)
from auto_disc.legacy.utils.misc.torch_utils import roll_n
from auto_disc.legacy.utils.spaces import BoxSpace, DictSpace, DiscreteSpace
from auto_disc.legacy.utils.spaces.utils import ConfigParameterBinding, distance

EPS = 0.0001
DISTANCE_WEIGHT = 2  # 1=linear, 2=quadratic, ...


def center_of_mass(input_array: torch.Tensor) -> torch.Tensor:
    normalizer = input_array.sum()
    grids = torch.meshgrid(*[torch.arange(0, i) for i in input_array.shape])

    center = torch.tensor(
        [
            (input_array * grids[dir].double()).sum() / normalizer
            for dir in range(input_array.ndim)
        ]
    )

    if torch.any(torch.isnan(center)):
        center = torch.tensor(
            [int((input_array.shape[0] - 1) / 2), int((input_array.shape[1] - 1) / 2)]
        )

    return center


def calc_distance_matrix(size_y: int, size_x: int) -> torch.Tensor:
    dist_mat = torch.zeros([size_y, size_x])

    mid_y = (size_y - 1) / 2
    mid_x = (size_x - 1) / 2
    mid = torch.tensor([mid_y, mid_x])

    max_dist = int(torch.linalg.norm(mid))

    for y in range(size_y):
        for x in range(size_x):
            dist_mat[y][x] = (
                1 - int(torch.linalg.norm(mid - torch.tensor([y, x]))) / max_dist
            ) ** DISTANCE_WEIGHT

    return dist_mat


def calc_image_moments(image: torch.Tensor) -> typing.Dict[str, torch.Tensor]:
    """
    Calculates the image moments for an image.

    For more information see:
     - https://en.wikipedia.org/wiki/Image_moment
     - http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/SHUTLER3/node1.html

    The code is based on the Javascript implementation of Lenia by Bert Chan.
    The code had to be adapted, because for the original code the coordinates are (x,y), whereas here they are (y,x).

    :param image: 2d gray scale image in form of a numpy array.
    :return: Namedtupel with the different moments.
    """

    eps = 0.00001

    size_y = image.shape[0]
    size_x = image.shape[1]

    x_grid, y_grid = torch.meshgrid(torch.arange(0, size_x), torch.arange(0, size_y))

    y_power1_image = y_grid * image
    y_power2_image = y_grid * y_power1_image
    y_power3_image = y_grid * y_power2_image

    x_power1_image = x_grid * image
    x_power2_image = x_grid * x_power1_image
    x_power3_image = x_grid * x_power2_image

    # raw moments: m_qp
    m00 = torch.sum(image)
    m10 = torch.sum(y_power1_image)
    m01 = torch.sum(x_power1_image)
    m11 = torch.sum(y_grid * x_grid * image)
    m20 = torch.sum(y_power2_image)
    m02 = torch.sum(x_power2_image)
    m21 = torch.sum(y_power2_image * x_grid)
    m12 = torch.sum(x_power2_image * y_grid)
    m22 = torch.sum(x_power2_image * y_grid * y_grid)
    m30 = torch.sum(y_power3_image)
    m31 = torch.sum(y_power3_image * x_grid)
    m13 = torch.sum(y_grid * x_power3_image)
    m03 = torch.sum(x_power3_image)
    m40 = torch.sum(y_power3_image * y_grid)
    m04 = torch.sum(x_power3_image * x_grid)

    # mY and mX describe the position of the centroid of the image
    # if there is no activation, then use the center position
    if m00 == 0:
        mY = (image.shape[0] - 1) / 2
        mX = (image.shape[1] - 1) / 2
    else:
        mY = m10 / m00
        mX = m01 / m00

    # in the case of very small activation (m00 ~ 0) the position becomes infinity, also then use the center position
    if mY == float("inf"):
        mY = (image.shape[0] - 1) / 2
    if mX == float("inf"):
        mX = (image.shape[1] - 1) / 2

    # calculate the central moments
    X2 = mX * mX
    X3 = X2 * mX
    Y2 = mY * mY
    Y3 = Y2 * mY
    XY = mX * mY

    mu11 = m11 - mY * m01
    mu20 = m20 - mY * m10
    mu02 = m02 - mX * m01
    mu30 = m30 - 3 * mY * m20 + 2 * Y2 * m10
    mu03 = m03 - 3 * mX * m02 + 2 * X2 * m01
    mu21 = m21 - 2 * mY * m11 - mX * m20 + 2 * Y2 * m01
    mu12 = m12 - 2 * mX * m11 - mY * m02 + 2 * X2 * m10
    mu22 = (
        m22
        - 2 * mX * m21
        + X2 * m20
        - 2 * mY * m12
        + 4 * XY * m11
        - 2 * mY * X2 * m10
        + Y2 * m02
        - 2 * Y2 * mX * m01
        + Y2 * X2 * m00
    )
    mu31 = (
        m31
        - mX * m30
        + 3 * mY * (mX * m20 - m21)
        + 3 * Y2 * (m11 - mX * m10)
        + Y3 * (mX * m00 - m01)
    )
    mu13 = (
        m13
        - mY * m03
        + 3 * mX * (mY * m02 - m12)
        + 3 * X2 * (m11 - mY * m01)
        + X3 * (mY * m00 - m10)
    )
    mu40 = m40 - 4 * mY * m30 + 6 * Y2 * m20 - 4 * Y3 * m10 + Y2 * Y2 * m00
    mu04 = m04 - 4 * mX * m03 + 6 * X2 * m02 - 4 * X3 * m01 + X2 * X2 * m00

    # Moment invariants: scale invariant
    if m00 < eps:
        eta11 = 0
        eta20 = 0
        eta02 = 0
        eta30 = 0
        eta03 = 0
        eta21 = 0
        eta12 = 0
        eta22 = 0
        eta31 = 0
        eta13 = 0
        eta40 = 0
        eta04 = 0
    else:
        m2 = m00 * m00
        mA = m00 * m00 * torch.sqrt(m00)
        m3 = m00 * m00 * m00
        eta11 = mu11 / m2
        eta20 = mu20 / m2
        eta02 = mu02 / m2
        eta30 = mu30 / mA
        eta03 = mu03 / mA
        eta21 = mu21 / mA
        eta12 = mu12 / mA
        eta22 = mu22 / m3
        eta31 = mu31 / m3
        eta13 = mu13 / m3
        eta40 = mu40 / m3
        eta04 = mu04 / m3

    # Moment invariants: rotation invariants
    Z = 2 * eta11
    A = eta20 + eta02
    B = eta20 - eta02
    C = eta30 + eta12
    D = eta30 - eta12
    E = eta03 + eta21
    F = eta03 - eta21
    G = eta30 - 3 * eta12
    H = 3 * eta21 - eta03
    Y = 2 * eta22
    I = eta40 + eta04
    J = eta40 - eta04
    K = eta31 + eta13
    L = eta31 - eta13
    CC = C * C
    EE = E * E
    CC_EE = CC - EE
    CC_EE3 = CC - 3 * EE
    CC3_EE = 3 * CC - EE
    CE = C * E
    DF = D * F
    M = I - 3 * Y
    t1 = CC_EE * CC_EE - 4 * CE * CE
    t2 = 4 * CE * CC_EE

    # invariants by Hu
    hu1 = A
    hu2 = B * B + Z * Z
    hu3 = G * G + H * H
    hu4 = CC + EE
    hu5 = G * C * CC_EE3 + H * E * CC3_EE
    hu6 = B * CC_EE + 2 * Z * CE
    hu7 = H * C * CC_EE3 - G * E * CC3_EE
    hu8 = Z * CC_EE / 2 - B * CE

    # extra invariants by Flusser
    flusser9 = I + Y
    flusser10 = J * CC_EE + 4 * L * DF
    flusser11 = -2 * K * CC_EE - 2 * J * DF
    flusser12 = 4 * L * t2 + M * t1
    flusser13 = -4 * L * t1 + M * t2

    result = Dict(
        y_avg=mY,
        x_avg=mX,
        m00=m00,
        m10=m10,
        m01=m01,
        m11=m11,
        m20=m20,
        m02=m02,
        m21=m21,
        m12=m12,
        m22=m22,
        m30=m30,
        m31=m31,
        m13=m13,
        m03=m03,
        m40=m40,
        m04=m04,
        mu11=mu11,
        mu20=mu20,
        mu02=mu02,
        mu30=mu30,
        mu03=mu03,
        mu21=mu21,
        mu12=mu12,
        mu22=mu22,
        mu31=mu31,
        mu13=mu13,
        mu40=mu40,
        mu04=mu04,
        eta11=eta11,
        eta20=eta20,
        eta02=eta02,
        eta30=eta30,
        eta03=eta03,
        eta21=eta21,
        eta12=eta12,
        eta22=eta22,
        eta31=eta31,
        eta13=eta13,
        eta40=eta40,
        eta04=eta04,
        hu1=hu1,
        hu2=hu2,
        hu3=hu3,
        hu4=hu4,
        hu5=hu5,
        hu6=hu6,
        hu7=hu7,
        hu8=hu8,
        flusser9=flusser9,
        flusser10=flusser10,
        flusser11=flusser11,
        flusser12=flusser12,
        flusser13=flusser13,
    )

    return result


@StringConfigParameter(name="distance_function", possible_values=["L2"], default="L2")
@IntegerConfigParameter(name="SX", default=256, min=1)
@IntegerConfigParameter(name="SY", default=256, min=1)
class LeniaHandDefinedRepresentation(BaseOutputRepresentation):
    CONFIG_DEFINITION = {}

    output_space = DictSpace(embedding=BoxSpace(low=0, high=0, shape=(17,)))

    def __init__(self, wrapped_input_space_key: str = None, **kwargs) -> None:
        super().__init__("states", **kwargs)

        # model
        self._statistic_names = [
            "activation_mass",
            "activation_volume",
            "activation_density",
            "activation_mass_distribution",
            "activation_hu1",
            "activation_hu2",
            "activation_hu3",
            "activation_hu4",
            "activation_hu5",
            "activation_hu6",
            "activation_hu7",
            "activation_hu8",
            "activation_flusser9",
            "activation_flusser10",
            "activation_flusser11",
            "activation_flusser12",
            "activation_flusser13",
        ]
        self._n_latents = len(self._statistic_names)

    def calc_static_statistics(self, final_obs: torch.Tensor) -> torch.Tensor:
        """Calculates the final statistics for lenia last observation"""

        feature_vector = torch.zeros(self._n_latents)
        cur_idx = 0

        size_y = self.config.SY
        size_x = self.config.SX
        num_of_cells = size_y * size_x

        # calc initial center of mass and use it as a reference point to "center" the world around it
        mid_y = (size_y - 1) / 2
        mid_x = (size_x - 1) / 2
        mid = torch.tensor([mid_y, mid_x])

        activation_center_of_mass = torch.tensor(center_of_mass(final_obs))
        activation_shift_to_center = mid - activation_center_of_mass

        activation = final_obs
        centered_activation = roll_n(activation, 0, activation_shift_to_center[0].int())
        centered_activation = roll_n(
            centered_activation, 1, activation_shift_to_center[1].int()
        )

        # calculate the image moments
        activation_moments = calc_image_moments(centered_activation)

        # activation mass
        activation_mass = activation_moments.m00
        # activation is number of acitvated cells divided by the number of cells
        activation_mass_data = activation_mass / num_of_cells
        feature_vector[cur_idx] = activation_mass_data
        cur_idx += 1

        # activation volume
        activation_volume = torch.sum(activation > EPS)
        activation_volume_data = activation_volume / num_of_cells
        feature_vector[cur_idx] = activation_volume_data
        cur_idx += 1

        # activation density
        if activation_volume == 0:
            activation_density_data = 0
        else:
            activation_density_data = activation_mass / activation_volume
        feature_vector[cur_idx] = activation_density_data
        cur_idx += 1

        # mass distribution around the center
        distance_weight_matrix = calc_distance_matrix(self.config.SY, self.config.SX)
        if activation_mass <= EPS:
            activation_mass_distribution = 1.0
        else:
            activation_mass_distribution = torch.sum(
                distance_weight_matrix * centered_activation
            ) / torch.sum(centered_activation)

        activation_mass_distribution_data = activation_mass_distribution
        feature_vector[cur_idx] = activation_mass_distribution_data
        cur_idx += 1

        # activation moments
        activation_hu1_data = activation_moments.hu1
        feature_vector[cur_idx] = activation_hu1_data
        cur_idx += 1

        activation_hu2_data = activation_moments.hu2
        feature_vector[cur_idx] = activation_hu2_data
        cur_idx += 1

        activation_hu3_data = activation_moments.hu3
        feature_vector[cur_idx] = activation_hu3_data
        cur_idx += 1

        activation_hu4_data = activation_moments.hu4
        feature_vector[cur_idx] = activation_hu4_data
        cur_idx += 1

        activation_hu5_data = activation_moments.hu5
        feature_vector[cur_idx] = activation_hu5_data
        cur_idx += 1

        activation_hu6_data = activation_moments.hu6
        feature_vector[cur_idx] = activation_hu6_data
        cur_idx += 1

        activation_hu7_data = activation_moments.hu7
        feature_vector[cur_idx] = activation_hu7_data
        cur_idx += 1

        activation_hu8_data = activation_moments.hu8
        feature_vector[cur_idx] = activation_hu8_data
        cur_idx += 1

        activation_flusser9_data = activation_moments.flusser9
        feature_vector[cur_idx] = activation_flusser9_data
        cur_idx += 1

        activation_flusser10_data = activation_moments.flusser10
        feature_vector[cur_idx] = activation_flusser10_data
        cur_idx += 1

        activation_flusser11_data = activation_moments.flusser11
        feature_vector[cur_idx] = activation_flusser11_data
        cur_idx += 1

        activation_flusser12_data = activation_moments.flusser12
        feature_vector[cur_idx] = activation_flusser12_data
        cur_idx += 1

        activation_flusser13_data = activation_moments.flusser13
        feature_vector[cur_idx] = activation_flusser13_data
        cur_idx += 1

        return feature_vector

    def map(
        self, input: Dict, is_output_new_discovery: bool
    ) -> typing.Dict[str, torch.Tensor]:
        """
        Compute statistics on Lenia's output
        Args:
            input: Lenia's output
            is_output_new_discovery: indicates if it is a new discovery
        Returns:
            Return a torch tensor in dict
        """

        embedding = self.calc_static_statistics(input.states[-1])

        return {"embedding": embedding}

    def calc_distance(
        self, embedding_a: torch.Tensor, embedding_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the distance between 2 embeddings in the latent space
        /!\ batch mode embedding_a and embedding_b can be N*M or M
        """
        # l2 loss
        if self.config.distance_function == "L2":
            dist = (embedding_a - embedding_b).pow(2).sum(-1).sqrt()

        else:
            raise NotImplementedError

        return dist
