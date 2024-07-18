import torch


def preprocess_data_ns(data, clifford):

    if clifford:
        # Concatenate zeros to the tensor
        zero_pad = torch.zeros(data.shape[:-1] + (1,))
        data = torch.cat((data, zero_pad), dim=-1)

    return data


def preprocess_data_maxwell3d(data, clifford):
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


def preprocess_data_maxwell2d(data, clifford):

    if clifford:

        # Concatenate zeros to the tensor
        zero_pad = torch.zeros(data.shape[:-1] + (1,))
        data = torch.cat(
            (zero_pad, zero_pad, zero_pad, zero_pad, data, zero_pad), dim=-1
        )

    return data


def preprocess_fn(experiment: str, clifford: bool):
    if experiment == "ns":
        return lambda x: preprocess_data_ns(x, clifford)
    elif experiment == "maxwell3d":
        return lambda x: preprocess_data_maxwell3d(x, clifford)
    elif experiment == "maxwell2d":
        return lambda x: preprocess_data_maxwell2d(x, clifford)
