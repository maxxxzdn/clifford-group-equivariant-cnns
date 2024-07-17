import torch


def preprocess_data_ns(data, clifford):

    if clifford:
        # Concatenate zeros to the tensor
        zero_pad = torch.zeros(data.shape[:-1] + (1,))
        data = torch.cat((data, zero_pad), dim=-1)

    return data


def preprocess_data_maxwell(data, clifford):
    preprocessing_factor = 10.0

    E_components = data[..., :3]  # First 3 components: Ex, Ey, Ez
    H_components = data[..., 3:]  # Last 3 components: Hx, Hy, Hz

    # Rearranging the magnetic field components: Hx, Hy, Hz -> Hz, -Hy, Hx
    H_components_rearranged = torch.stack(
        [
            H_components[..., 2],  # Hz
            -H_components[..., 1],  # -Hy
            H_components[..., 0],  # Hx
        ],
        dim=-1,
    )

    # Concatenating the electric and rearranged magnetic components
    data = torch.cat([E_components, H_components_rearranged], dim=-1)

    if clifford:
        # Concatenate zeros to the tensor
        zero_pad = torch.zeros(data.shape[:-1] + (1,))
        data = torch.cat((zero_pad, data, zero_pad), dim=-1)

    return data * preprocessing_factor


def preprocess_data_maxwell2d_highres(data, clifford):

    # data = F.resize(data.permute(0, 3, 1, 2), size=[128,128]).permute(0, 2, 3, 1)

    if clifford:

        # Concatenate zeros to the tensor
        zero_pad = torch.zeros(data.shape[:-1] + (1,))
        data = torch.cat(
            (zero_pad, zero_pad, zero_pad, zero_pad, data, zero_pad), dim=-1
        )

    return data


# mean = torch.tensor([[0.0003]]).reshape(1,1,1,1) # 0.0005
# std = torch.tensor([0.0275]).reshape(1,1,1,1) # 0.0282


def preprocess_data_maxwell2d(data, clifford):

    if clifford:

        # Concatenate zeros to the tensor
        zero_pad = torch.zeros(data.shape[:-1] + (1,))
        data = torch.cat(
            (zero_pad, zero_pad, zero_pad, zero_pad, data, zero_pad), dim=-1
        )

    return data


def preprocess_fn(experiment: str, clifford: bool, highres: bool = False):
    if experiment == "ns":
        return lambda x: preprocess_data_ns(x, clifford)
    elif experiment == "maxwell":
        return lambda x: preprocess_data_maxwell(x, clifford)
    elif experiment == "maxwell2d":
        if highres:
            return lambda x: preprocess_data_maxwell2d_highres(x, clifford)
        else:
            return lambda x: preprocess_data_maxwell2d(x, clifford)
