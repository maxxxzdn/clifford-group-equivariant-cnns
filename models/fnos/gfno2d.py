import torch.nn.functional as F
import torch
import torch.nn as nn
import math


class grid(torch.nn.Module):
    def __init__(self, twoD, grid_type):
        super(grid, self).__init__()
        assert grid_type in ["cartesian", "symmetric", "None"], "Invalid grid type"
        self.symmetric = grid_type == "symmetric"
        self.include_grid = grid_type != "None"
        self.grid_dim = (1 + (not self.symmetric) + (not twoD)) * self.include_grid
        if self.include_grid:
            if twoD:
                self.get_grid = self.twoD_grid
            else:
                self.get_grid = self.threeD_grid
        else:
            self.get_grid = torch.nn.Identity()

    def forward(self, x):
        return self.get_grid(x)

    def twoD_grid(self, x):
        shape = x.shape
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = (
            torch.linspace(0, 1, size_x)
            .reshape(1, size_x, 1, 1)
            .repeat([batchsize, 1, size_y, 1])
        )
        gridy = (
            torch.linspace(0, 1, size_y)
            .reshape(1, 1, size_y, 1)
            .repeat([batchsize, size_x, 1, 1])
        )
        if not self.symmetric:
            grid = torch.cat((gridx, gridy), dim=-1)
        else:
            midx = 0.5
            midy = (size_y - 1) / (2 * (size_x - 1))
            gridx = (gridx - midx) ** 2
            gridy = (gridy - midy) ** 2
            grid = gridx + gridy
        grid = grid.to(x.device)
        return torch.cat((x, grid), dim=-1)

    def threeD_grid(self, x):
        shape = x.shape
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = (
            torch.linspace(0, 1, size_x)
            .reshape(1, size_x, 1, 1, 1)
            .repeat([batchsize, 1, size_y, size_z, 1])
        )
        gridy = (
            torch.linspace(0, 1, size_y)
            .reshape(1, 1, size_y, 1, 1)
            .repeat([batchsize, size_x, 1, size_z, 1])
        )
        gridz = (
            torch.linspace(0, 1, size_z)
            .reshape(1, 1, 1, size_z, 1)
            .repeat([batchsize, size_x, size_y, 1, 1])
        )
        if not self.symmetric:
            grid = torch.cat((gridx, gridy, gridz), dim=-1)
        else:
            midx = 0.5
            midy = (size_y - 1) / (2 * (size_x - 1))
            gridx = (gridx - midx) ** 2
            gridy = (gridy - midy) ** 2
            grid = torch.cat((gridx + gridy, gridz), dim=-1)
        grid = grid.to(x.device)
        return torch.cat((x, grid), dim=-1)


# ----------------------------------------------------------------------------------------------------------------------
# GFNO2d
# ----------------------------------------------------------------------------------------------------------------------
class GConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        bias=True,
        first_layer=False,
        last_layer=False,
        spectral=False,
        Hermitian=False,
        reflection=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reflection = reflection
        self.rt_group_size = 4
        self.group_size = self.rt_group_size * (1 + reflection)
        assert kernel_size % 2 == 1, "kernel size must be odd"
        dtype = torch.cfloat if spectral else torch.float
        self.kernel_size_Y = kernel_size
        self.kernel_size_X = kernel_size // 2 + 1 if Hermitian else kernel_size
        self.Hermitian = Hermitian
        if first_layer or last_layer:
            self.W = nn.Parameter(
                torch.empty(
                    out_channels,
                    1,
                    in_channels,
                    self.kernel_size_Y,
                    self.kernel_size_X,
                    dtype=dtype,
                )
            )
        else:
            if self.Hermitian:
                self.W = nn.ParameterDict(
                    {
                        "y0_modes": torch.nn.Parameter(
                            torch.empty(
                                out_channels,
                                1,
                                in_channels,
                                self.group_size,
                                self.kernel_size_X - 1,
                                1,
                                dtype=dtype,
                            )
                        ),
                        "yposx_modes": torch.nn.Parameter(
                            torch.empty(
                                out_channels,
                                1,
                                in_channels,
                                self.group_size,
                                self.kernel_size_Y,
                                self.kernel_size_X - 1,
                                dtype=dtype,
                            )
                        ),
                        "00_modes": torch.nn.Parameter(
                            torch.empty(
                                out_channels,
                                1,
                                in_channels,
                                self.group_size,
                                1,
                                1,
                                dtype=torch.float,
                            )
                        ),
                    }
                )
            else:
                self.W = nn.Parameter(
                    torch.empty(
                        out_channels,
                        1,
                        in_channels,
                        self.group_size,
                        self.kernel_size_Y,
                        self.kernel_size_X,
                        dtype=dtype,
                    )
                )
        self.first_layer = first_layer
        self.last_layer = last_layer
        self.B = nn.Parameter(torch.empty(1, out_channels, 1, 1)) if bias else None
        self.eval_build = True
        self.reset_parameters()
        self.get_weight()

    def reset_parameters(self):
        if self.Hermitian:
            for v in self.W.values():
                nn.init.kaiming_uniform_(v, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.B is not None:
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def get_weight(self):

        if self.training:
            self.eval_build = True
        elif self.eval_build:
            self.eval_build = False
        else:
            return

        if self.Hermitian:
            self.weights = torch.cat(
                [
                    self.W["y0_modes"],
                    self.W["00_modes"].cfloat(),
                    self.W["y0_modes"].flip(dims=(-2,)).conj(),
                ],
                dim=-2,
            )
            self.weights = torch.cat([self.weights, self.W["yposx_modes"]], dim=-1)
            self.weights = torch.cat(
                [self.weights[..., 1:].conj().rot90(k=2, dims=[-2, -1]), self.weights],
                dim=-1,
            )
        else:
            self.weights = self.W[:]

        if self.first_layer or self.last_layer:

            # construct the weight
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1)

            # apply each of the group elements to the corresponding repetition
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k].rot90(k=k, dims=[-2, -1])

            # apply each the reflection group element to the rotated kernels
            if self.reflection:
                self.weights[:, self.rt_group_size :] = self.weights[
                    :, : self.rt_group_size
                ].flip(dims=[-2])

            # collapse out_channels and group1 dimensions for use with conv2d
            if self.first_layer:
                self.weights = self.weights.view(
                    -1, self.in_channels, self.kernel_size_Y, self.kernel_size_Y
                )
                if self.B is not None:
                    self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)
            else:
                self.weights = self.weights.transpose(2, 1).reshape(
                    self.out_channels, -1, self.kernel_size_Y, self.kernel_size_Y
                )
                self.bias = self.B

        else:

            # construct the weight
            self.weights = self.weights.repeat(1, self.group_size, 1, 1, 1, 1)

            # apply elements in the rotation group
            for k in range(1, self.rt_group_size):
                self.weights[:, k] = self.weights[:, k - 1].rot90(dims=[-2, -1])

                if self.reflection:
                    self.weights[:, k] = torch.cat(
                        [
                            self.weights[:, k, :, self.rt_group_size - 1].unsqueeze(2),
                            self.weights[:, k, :, : (self.rt_group_size - 1)],
                            self.weights[:, k, :, (self.rt_group_size + 1) :],
                            self.weights[:, k, :, self.rt_group_size].unsqueeze(2),
                        ],
                        dim=2,
                    )
                else:
                    self.weights[:, k] = torch.cat(
                        [
                            self.weights[:, k, :, -1].unsqueeze(2),
                            self.weights[:, k, :, :-1],
                        ],
                        dim=2,
                    )

            if self.reflection:
                # apply elements in the reflection group
                self.weights[:, self.rt_group_size :] = torch.cat(
                    [
                        self.weights[:, : self.rt_group_size, :, self.rt_group_size :],
                        self.weights[:, : self.rt_group_size, :, : self.rt_group_size],
                    ],
                    dim=3,
                ).flip([-2])

            # collapse out_channels / groups1 and in_channels/groups2 dimensions for use with conv2d
            self.weights = self.weights.view(
                self.out_channels * self.group_size,
                self.in_channels * self.group_size,
                self.kernel_size_Y,
                self.kernel_size_Y,
            )
            if self.B is not None:
                self.bias = self.B.repeat_interleave(repeats=self.group_size, dim=1)

        if self.Hermitian:
            self.weights = self.weights[..., -self.kernel_size_X :]

    def forward(self, x):

        self.get_weight()

        # output is of shape (batch * out_channels, number of group elements, ny, nx)
        x = nn.functional.conv2d(input=x, weight=self.weights)

        # add the bias
        if self.B is not None:
            x = x + self.bias
        return x


class GSpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, reflection=False):
        super(GSpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = (
            modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.conv = GConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2 * modes - 1,
            reflection=reflection,
            bias=False,
            spectral=True,
            Hermitian=True,
        )
        self.get_weight()

    # Building the weight
    def get_weight(self):
        self.conv.get_weight()
        self.weights = self.conv.weights.transpose(0, 1)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]

        # get the index of the zero frequency and construct weight
        freq0_y = (
            (torch.fft.fftshift(torch.fft.fftfreq(x.shape[-2])) == 0).nonzero().item()
        )
        self.get_weight()

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.fftshift(torch.fft.rfft2(x), dim=-2)
        x_ft = x_ft[
            ..., (freq0_y - self.modes + 1) : (freq0_y + self.modes), : self.modes
        ]

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.weights.shape[0],
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[
            ..., (freq0_y - self.modes + 1) : (freq0_y + self.modes), : self.modes
        ] = self.compl_mul2d(x_ft, self.weights)

        # Return to physical space
        x = torch.fft.irfft2(
            torch.fft.ifftshift(out_ft, dim=-2), s=(x.size(-2), x.size(-1))
        )

        return x


class GMLP2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels,
        reflection=False,
        last_layer=False,
    ):
        super(GMLP2d, self).__init__()
        self.mlp1 = GConv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            reflection=reflection,
        )
        self.mlp2 = GConv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            reflection=reflection,
            last_layer=last_layer,
        )

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class MLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP2d, self).__init__()
        self.mlp1 = nn.Conv2d(
            in_channels=in_channels, out_channels=mid_channels, kernel_size=1
        )
        self.mlp2 = nn.Conv2d(
            in_channels=mid_channels, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class GNorm(nn.Module):
    def __init__(self, width, group_size):
        super().__init__()
        self.group_size = group_size
        self.norm = torch.nn.InstanceNorm3d(width)

    def forward(self, x):
        x = x.view(x.shape[0], -1, self.group_size, x.shape[-2], x.shape[-1])
        x = self.norm(x)
        x = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1])
        return x


class _GFNO2d(nn.Module):
    def __init__(
        self,
        num_channels,
        modes,
        width,
        initial_step,
        reflection=True,
        grid_type="cartesian",
        noneq_proj=False,
    ):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes = modes
        self.width = width
        self.noneq_proj = noneq_proj
        self.grid = grid(twoD=True, grid_type=grid_type)

        if noneq_proj:
            self.p = nn.Conv2d(
                in_channels=num_channels * initial_step + self.grid.grid_dim,
                out_channels=self.width * 8,
                kernel_size=1,
            )
        else:
            self.p = GConv2d(
                in_channels=num_channels * initial_step + self.grid.grid_dim,
                out_channels=self.width,
                kernel_size=1,
                reflection=reflection,
                first_layer=True,
            )
        self.conv0 = GSpectralConv2d(
            in_channels=self.width,
            out_channels=self.width,
            modes=self.modes,
            reflection=reflection,
        )
        self.conv1 = GSpectralConv2d(
            in_channels=self.width,
            out_channels=self.width,
            modes=self.modes,
            reflection=reflection,
        )
        self.conv2 = GSpectralConv2d(
            in_channels=self.width,
            out_channels=self.width,
            modes=self.modes,
            reflection=reflection,
        )
        self.conv3 = GSpectralConv2d(
            in_channels=self.width,
            out_channels=self.width,
            modes=self.modes,
            reflection=reflection,
        )
        self.mlp0 = GMLP2d(
            in_channels=self.width,
            out_channels=self.width,
            mid_channels=self.width,
            reflection=reflection,
        )
        self.mlp1 = GMLP2d(
            in_channels=self.width,
            out_channels=self.width,
            mid_channels=self.width,
            reflection=reflection,
        )
        self.mlp2 = GMLP2d(
            in_channels=self.width,
            out_channels=self.width,
            mid_channels=self.width,
            reflection=reflection,
        )
        self.mlp3 = GMLP2d(
            in_channels=self.width,
            out_channels=self.width,
            mid_channels=self.width,
            reflection=reflection,
        )
        self.w0 = GConv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=1,
            reflection=reflection,
        )
        self.w1 = GConv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=1,
            reflection=reflection,
        )
        self.w2 = GConv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=1,
            reflection=reflection,
        )
        self.w3 = GConv2d(
            in_channels=self.width,
            out_channels=self.width,
            kernel_size=1,
            reflection=reflection,
        )
        self.norm = GNorm(self.width, group_size=4 * (1 + reflection))
        if noneq_proj:
            self.q = MLP2d(
                in_channels=self.width * 8,
                mid_channels=self.width * 4,
                out_channels=num_channels,
            )
        else:
            self.q = GMLP2d(
                in_channels=self.width,
                out_channels=num_channels,
                mid_channels=self.width * 4,
                reflection=reflection,
                last_layer=True,
            )  # output channel is 1: u(x, y)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], x.shape[2], -1)
        x = self.grid(x)
        x = x.permute(0, 3, 1, 2)
        x = self.p(x)
        x1 = self.norm(self.conv0(self.norm(x)))
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv1(self.norm(x)))
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv2(self.norm(x)))
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.norm(self.conv3(self.norm(x)))
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x.unsqueeze(-2)


class GFNO2d(nn.Module):
    def __init__(
        self, time_history, time_future, channels, modes, hidden_channels, noneq_proj
    ):
        super(GFNO2d, self).__init__()
        self.time_future = time_future
        self.fno = _GFNO2d(
            channels, modes, hidden_channels, time_history, noneq_proj=noneq_proj
        )

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.permute(0, 2, 3, 1, 4)  # (B,...,T,C)
        x = x.reshape(B, H, W, -1)  # (B,...,T*C)

        x = self.fno(x)  # (B,...,T*C)

        x = x.reshape(B, H, W, self.time_future, C)
        x = x.permute(0, 3, 1, 2, 4)
        return x
