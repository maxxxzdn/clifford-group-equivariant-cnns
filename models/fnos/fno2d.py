from torch import nn
from neuralop.models import FNO


class FNO2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        modes1,
        modes2,
        projection_channels,
    ):
        super(FNO2d, self).__init__()

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
        self.out_channels = out_channels
        self.fno = FNO(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=(modes1, modes2),
            hidden_channels=hidden_channels,
            projection_channels=projection_channels,
        )

    def forward(self, x):
        B, T, H, W, C = x.shape
        x = x.permute(0, 2, 3, 1, 4)  # (B,...,T,C)
        x = x.reshape(B, H, W, -1)  # (B,...,T*C)
        x = x.permute(0, 3, 1, 2)  # (B,T*C,...)

        x = self.fno(x)  # (B,T*C,...)

        T_out = self.out_channels // C
        x = x.reshape(B, T_out, C, H, W)
        x = x.permute(0, 1, 3, 4, 2)
        return x
