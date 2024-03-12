import torch
from torch import nn

PI = torch.acos(torch.zeros(1)).item() * 2


class SphericPad(nn.Module):
    """Pads spherically the input on all sides with the given padding size."""

    def __init__(self, padding_size: torch.Tensor) -> None:
        super(SphericPad, self).__init__()
        if isinstance(padding_size, int) or (
            isinstance(padding_size, torch.Tensor) and padding_size.shape == ()
        ):
            self.pad_left = (
                self.pad_right
            ) = self.pad_top = self.pad_bottom = padding_size
        elif (
            isinstance(padding_size, tuple) or isinstance(padding_size, torch.Tensor)
        ) and len(padding_size) == 2:
            self.pad_left = self.pad_right = padding_size[0]
            self.pad_top = self.pad_bottom = padding_size[1]
        elif (
            isinstance(padding_size, tuple) or isinstance(padding_size, torch.Tensor)
        ) and len(padding_size) == 4:
            self.pad_left = padding_size[0]
            self.pad_top = padding_size[1]
            self.pad_right = padding_size[2]
            self.pad_bottom = padding_size[3]
        else:
            raise ValueError(
                "The padding size shoud be: int, torch.IntTensor  or tuple of size 2 or tuple of size 4"
            )

    def forward(self, input):
        output = torch.cat(
            [input, input[:, :, : int(self.pad_bottom.item()), :]], dim=2
        )
        output = torch.cat(
            [output, output[:, :, :, : int(self.pad_right.item())]], dim=3
        )
        output = torch.cat(
            [
                output[
                    :,
                    :,
                    -(int((self.pad_bottom + self.pad_top).item())) : -int(
                        self.pad_bottom.item()
                    ),
                    :,
                ],
                output,
            ],
            dim=2,
        )
        output = torch.cat(
            [
                output[
                    :,
                    :,
                    :,
                    -(int((self.pad_right + self.pad_left).item())) : -int(
                        self.pad_right.item()
                    ),
                ],
                output,
            ],
            dim=3,
        )

        return output


def complex_mult_torch(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Computes the complex multiplication in Pytorch when the tensor last dimension is 2: 0 is the real component and 1 the imaginary one"""
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, "Last dimension must be 2"
    return torch.stack(
        (
            X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1],
            X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0],
        ),
        dim=-1,
    )


def roll_n(X: torch.Tensor, axis: int, n: int) -> torch.Tensor:
    """Rolls a tensor with a shift n on the specified axis"""
    f_idx = tuple(
        slice(None, None, None) if i != axis else slice(0, n, None)
        for i in range(X.dim())
    )
    b_idx = tuple(
        slice(None, None, None) if i != axis else slice(n, None, None)
        for i in range(X.dim())
    )
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def soft_max(x, m, k):
    return torch.log(torch.exp(k * x) + torch.exp(k * m)) / k


def soft_clip(x, min, max, k):
    a = torch.exp(k * x)
    b = torch.exp(torch.FloatTensor([k * min])).item()
    c = torch.exp(torch.FloatTensor([-k * max])).item()
    return torch.log(1.0 / (a + b) + c) / -k
