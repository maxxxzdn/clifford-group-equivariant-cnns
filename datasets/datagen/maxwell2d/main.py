import warnings

warnings.filterwarnings(
    "error",
    category=RuntimeWarning,
    message="some failed to converge after 50 iterations",
)

import torch
import torch
import torchvision.transforms.functional as F

torch.set_default_dtype(torch.float64)

import numpy as np
from scipy import constants
from tqdm import tqdm

import pycharge as pc
from pycharge.charges import *

from utils.algebra import CliffordAlgebra


# Constants
c = constants.c
e = constants.e


class OscillatingChargeMoving(Charge):
    """Sinusoidally oscillating charge along a specified axis.

    Args:
        origin (Tuple[float, float, float]): List of x, y, and z values for
            the oscillating charge's origin.
        direction (Tuple[float, float, float]): List of x, y, and z values
            for the charge direction vector.
        amplitude (float): Amplitude of the oscillations.
        omega (float): Angular frequency of the oscillations (units: rad/s).
        start_zero (bool): Determines if the charge begins oscillating at t=0.
            Defaults to `False`.
        stop_t (Optional[float]): Time when the charge stops oscillating.
            If `None`, the charge never stops oscillating. Defaults to `None`.
        q (float): Charge value, can be positive or negative. Default is `e`.
    """

    def __init__(
        self,
        origin: Tuple[float, float, float],
        const_v: Tuple[
            float, float, float
        ],  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        direction: Tuple[float, float, float],
        amplitude: float,
        omega: float,
        start_zero: bool = False,
        stop_t: Optional[float] = None,
        q: float = e,
    ) -> None:
        super().__init__(q)
        self.origin = np.array(origin)
        self.const_v = np.array(
            const_v
        )  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.direction = np.array(direction) / np.linalg.norm(np.array(direction))
        self.amplitude = amplitude
        self.omega = omega
        self.start_zero = start_zero
        self.stop_t = stop_t

    def xpos(self, t: Union[ndarray, float]) -> ndarray:
        xpos = (
            self.direction[0] * self.amplitude * np.cos(self.omega * t)
            + self.const_v[0]
            * t  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            + self.origin[0]
        )
        if self.start_zero:
            xpos[t < 0] = self.origin[0]
        if self.stop_t is not None:
            xpos[t > self.stop_t] = (
                self.direction[0] * self.amplitude * np.cos(self.omega * self.stop_t)
                + self.origin[0]
            )
        return xpos

    def ypos(self, t: Union[ndarray, float]) -> ndarray:
        ypos = (
            self.direction[1] * self.amplitude * np.cos(self.omega * t)
            + self.const_v[1]
            * t  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            + self.origin[1]
        )
        if self.start_zero:
            ypos[t < 0] = self.origin[1]
        if self.stop_t is not None:
            ypos[t > self.stop_t] = (
                self.direction[1] * self.amplitude * np.cos(self.omega * self.stop_t)
                + self.origin[1]
            )
        return ypos

    def zpos(self, t: Union[ndarray, float]) -> ndarray:
        zpos = (
            self.direction[2] * self.amplitude * np.cos(self.omega * t)
            + self.const_v[2]
            * t  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            + self.origin[2]
        )
        if self.start_zero:
            zpos[t < 0] = self.origin[2]
        if self.stop_t is not None:
            zpos[t > self.stop_t] = (
                self.direction[2] * self.amplitude * np.cos(self.omega * self.stop_t)
                + self.origin[2]
            )
        return zpos

    def xvel(self, t: Union[ndarray, float]) -> ndarray:
        xvel = -(
            (
                self.direction[0]
                * self.amplitude  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                * self.omega
                * np.sin(self.omega * t)
            )
            + self.const_v[
                0
            ]  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        )  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        if self.start_zero:
            xvel[t < 0] = 0
        if self.stop_t is not None:
            xvel[t > self.stop_t] = 0
        return xvel

    def yvel(self, t: Union[ndarray, float]) -> ndarray:
        yvel = -(
            (
                self.direction[1]
                * self.amplitude  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                * self.omega
                * np.sin(self.omega * t)
            )
            + self.const_v[
                1
            ]  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        )  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        if self.start_zero:
            yvel[t < 0] = 0
        if self.stop_t is not None:
            yvel[t > self.stop_t] = 0
        return yvel

    def zvel(self, t: Union[ndarray, float]) -> ndarray:
        zvel = -(
            (
                self.direction[2]
                * self.amplitude  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                * self.omega
                * np.sin(self.omega * t)
            )
            + self.const_v[
                2
            ]  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        )  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        if self.start_zero:
            zvel[t < 0] = 0
        if self.stop_t is not None:
            zvel[t > self.stop_t] = 0
        return zvel

    def xacc(self, t: Union[ndarray, float]) -> ndarray:
        xacc = -(
            self.direction[0] * self.amplitude * self.omega**2 * np.cos(self.omega * t)
        )
        if self.start_zero:
            xacc[t < 0] = 0
        if self.stop_t is not None:
            xacc[t > self.stop_t] = 0
        return xacc

    def yacc(self, t: Union[ndarray, float]) -> ndarray:
        yacc = -(
            self.direction[1] * self.amplitude * self.omega**2 * np.cos(self.omega * t)
        )
        if self.start_zero:
            yacc[t < 0] = 0
        if self.stop_t is not None:
            yacc[t > self.stop_t] = 0
        return yacc

    def zacc(self, t: Union[ndarray, float]) -> ndarray:
        zacc = -(
            self.direction[2] * self.amplitude * self.omega**2 * np.cos(self.omega * t)
        )
        if self.start_zero:
            zacc[t < 0] = 0
        if self.stop_t is not None:
            zacc[t > self.stop_t] = 0
        return zacc


class OrbittingChargeMoving(Charge):
    """Radially orbitting charge in the x-y plane.

    At t=0, the charge is at the position (x=`radius`, y=0, z=0) and orbits
    counter-clockwise.

    Args:
        radius (float): Radius of the orbitting charge trajectory.
        omega (float): Angular frequency of the orbit (units: rad/s).
        start_zero (bool): Determines if the charge begins orbitting at t=0.
            Defaults to `False`.
        stop_t (Optional[float]): Time when the charge stops orbitting.
            If `None`, the charge never stops orbitting. Defaults to `None`.
        q (float): Charge value, can be positive or negative. Default is `e`.
    """

    def __init__(
        self,
        origin: Tuple[float, float, float],
        radius: float,
        omega: float,
        const_v: Tuple[
            float, float, float
        ],  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        start_zero: bool = False,
        stop_t: Optional[float] = None,
        q: float = e,
    ) -> None:
        super().__init__(q)
        self.origin = np.array(origin)
        self.radius = radius
        self.omega = omega
        self.const_v = np.array(const_v)
        self.start_zero = start_zero
        self.stop_t = stop_t

    def xpos(self, t: Union[ndarray, float]) -> ndarray:
        xpos = self.radius * np.cos(self.omega * t)
        if self.start_zero:
            xpos[t < 0] = self.radius
        if self.stop_t is not None:
            xpos[t > self.stop_t] = self.radius * np.cos(self.omega * self.stop_t)
        return xpos + self.const_v[0] * t

    def ypos(self, t: Union[ndarray, float]) -> ndarray:
        ypos = self.radius * np.sin(self.omega * t)
        if self.start_zero:
            ypos[t < 0] = 0
        if self.stop_t is not None:
            ypos[t > self.stop_t] = self.radius * np.sin(self.omega * self.stop_t)
        return ypos + self.const_v[1] * t

    def zpos(self, t: Union[ndarray, float]) -> float:
        return 0

    def xvel(self, t: Union[ndarray, float]) -> ndarray:
        xvel = -self.radius * self.omega * np.sin(self.omega * t)
        if self.start_zero:
            xvel[t < 0] = 0
        if self.stop_t is not None:
            xvel[t > self.stop_t] = 0
        return xvel + self.const_v[0]

    def yvel(self, t: Union[ndarray, float]) -> ndarray:
        yvel = self.radius * self.omega * np.cos(self.omega * t)
        if self.start_zero:
            yvel[t < 0] = 0
        if self.stop_t is not None:
            yvel[t > self.stop_t] = 0
        return yvel + self.const_v[1]

    def zvel(self, t: Union[ndarray, float]) -> float:
        return 0

    def xacc(self, t: Union[ndarray, float]) -> ndarray:
        xacc = -self.radius * self.omega**2 * np.cos(self.omega * t)
        if self.start_zero:
            xacc[t < 0] = 0
        if self.stop_t is not None:
            xacc[t > self.stop_t] = 0
        return xacc

    def yacc(self, t: Union[ndarray, float]) -> ndarray:
        yacc = -self.radius * self.omega**2 * np.sin(self.omega * t)
        if self.start_zero:
            yacc[t < 0] = 0
        if self.stop_t is not None:
            yacc[t > self.stop_t] = 0
        return yacc

    def zacc(self, t: Union[ndarray, float]) -> float:
        return 0


# grid geometry
LIM = 50e-9  # *half* of the grid extension
N_FRAMES = 256
GRID_SIZE = 256
GRID_OUT_SIZE = 32
GRID_SIZE_QUIVER = 17
LIM_QUIVER = 46e-9

ORIGIN_LIM = 0.3 * LIM  # initial box size in which sources are sampled
ORIGIN_DIST = 0.1 * LIM  # forced initial distance between charges

# sources / charges
N_OSC_MIN, N_OSC_MAX = 3, 4
N_ORB_MIN, N_ORB_MAX = 5, 7
CHARGE_VALS = np.array([3, 2, 1, -1, -2, -3]) * e
BOOST_FACTOR = 0.5

# velocity parameters
V_MAX = 0.5 * c  # maximum stable value
V_FRAC_CONST = 0.4  # fraction of total velocity going into boosts, the rest goes into oscillations/orbiting
V_FRAC_REL = 0.4  # fraction of constant velocity going into relative vs. absolute motion of sources

# oscillating
# need to ensure that max velocity  =  OSC_AMP_MAX * OSC_OMEGA_MAX  <=  (1-V_FRAC_CONST) * V_MAX
OSC_AMP_MIN = 0.02 * LIM
OSC_AMP_MAX = 0.04 * LIM
OSC_OMEGA_MIN = 0.67 * (1 - V_FRAC_CONST) * V_MAX / OSC_AMP_MAX
OSC_OMEGA_MAX = (1 - V_FRAC_CONST) * V_MAX / OSC_AMP_MAX

# orbiting
# need to ensure that max velocity  =  ORB_RADIUS_MAX * ORB_OMEGA_MAX  <=  (1-V_FRAC_CONST) * V_MAX
ORB_RADIUS_MIN = 0.025 * LIM
ORB_RADIUS_MAX = 0.05 * LIM
ORB_OMEGA_MIN = 0.75 * (1 - V_FRAC_CONST) * V_MAX / ORB_RADIUS_MAX
ORB_OMEGA_MAX = (1 - V_FRAC_CONST) * V_MAX / ORB_RADIUS_MAX


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def sample_unit_vectors(N):
    vectors = np.random.normal(size=(N, 3))
    vectors[:, -1] = 0  # living in xy-plane
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors


def gen_sources():

    ################################################################################
    # sample parameters
    N_osc = np.random.randint(N_OSC_MIN, N_OSC_MAX + 1)
    N_orb = np.random.randint(N_ORB_MIN, N_ORB_MAX + 1)
    charges = np.random.choice(CHARGE_VALS, size=N_osc + N_orb)

    origins = []
    while len(origins) < N_osc + N_orb:
        origin = np.random.uniform(-ORIGIN_LIM, ORIGIN_LIM, size=3)
        origin[2] = 0
        if all(distance(origin, existing) >= ORIGIN_DIST for existing in origins):
            origins.append(origin)

    v_abs = V_MAX * V_FRAC_CONST * (1 - V_FRAC_REL) * sample_unit_vectors(1)[0]
    v_rel = V_MAX * V_FRAC_CONST * V_FRAC_REL * sample_unit_vectors(N_osc + N_orb)

    osc_directions = sample_unit_vectors(N_osc)
    osc_amplitudes = np.random.uniform(OSC_AMP_MIN, OSC_AMP_MAX, size=N_osc)
    osc_omegas = np.random.uniform(OSC_OMEGA_MIN, OSC_OMEGA_MAX, size=N_osc)

    orb_radii = np.random.uniform(ORB_RADIUS_MIN, ORB_RADIUS_MAX, size=N_orb)
    orb_omegas = np.random.uniform(ORB_OMEGA_MIN, ORB_OMEGA_MAX, size=N_orb)

    ################################################################################
    # generate sources
    sources = []
    osc_params = []
    orb_params = []

    for i in range(N_osc):
        osc_params.append(
            {
                "origin": origins[i],
                "const_v": (v_abs + v_rel[i]) * BOOST_FACTOR,
                "q": charges[i],
                "direction": osc_directions[i],
                "amplitude": osc_amplitudes[i],
                "omega": osc_omegas[i],
            }
        )
        sources.append(OscillatingChargeMoving(**osc_params[-1]))

    for i in range(N_orb):
        orb_params.append(
            {
                "origin": origins[i + N_osc],
                "const_v": (v_abs + v_rel[i + N_osc]) * BOOST_FACTOR,
                "q": charges[i + N_osc],
                "radius": orb_radii[i],
                "omega": orb_omegas[i],
            }
        )
        sources.append(OrbittingChargeMoving(**orb_params[-1]))

    return sources, osc_params, orb_params


def simulate():
    sources, osc_params, orb_params = gen_sources()
    simulation = pc.Simulation(sources)

    x, y, z = np.meshgrid(
        np.linspace(-LIM, LIM, GRID_SIZE),
        np.linspace(-LIM, LIM, GRID_SIZE),
        0,
        indexing="ij",
    )

    dt = 2 * np.pi / 7.5e16 / N_FRAMES * 8

    U_store = np.empty((3, N_FRAMES, GRID_SIZE, GRID_SIZE), dtype=np.float64)

    for frame in tqdm(range(N_FRAMES)):
        t = frame * dt
        E_total = simulation.calculate_E(t=t, x=x, y=y, z=z, pcharge_field="Total")
        B_total = simulation.calculate_B(t=t, x=x, y=y, z=z, pcharge_field="Total")

        U_store[0, frame] = E_total[0][:, :, 0]
        U_store[1, frame] = E_total[1][:, :, 0]
        U_store[2, frame] = B_total[2][:, :, 0] * c

    U_store = np.transpose(U_store, (1, 2, 3, 0))

    ### NORMALIZE DATA ###
    cl = CliffordAlgebra((-1, 1, 1))
    d_cl = cl.embed_grade(torch.from_numpy(U_store), 2)
    d_norm = cl.norm(d_cl, safe_abs_sqrt=False)

    ### PREPROCESSING ###
    d_norm_ = torch.clip(d_norm, 1e7, None)
    d_norm_cl = torch.log10(d_norm) * d_cl / d_norm_

    norm_min, norm_max = d_norm_cl.min(), d_norm_cl.max()
    d_cl_min, d_cl_max = d_cl.min(), d_cl.max()

    data_params = {
        "data_min": d_cl_min.item(),
        "data_max": d_cl_max.item(),
        "norm_min": norm_min.item(),
        "norm_max": norm_max.item(),
    }

    # UPDATE TO NORMED
    U_store = d_norm_cl[..., [4, 5, 6]]  # take only E_x, E_y, B_z components

    # resize to GRID_OUT_SIZE
    U_store = F.resize(
        U_store.permute(0, 3, 1, 2), size=[GRID_OUT_SIZE, GRID_OUT_SIZE]
    ).permute(0, 2, 3, 1)

    return U_store, data_params, osc_params, orb_params


def main(args):

    # try generating data until a valid configuration is found
    while True:
        try:
            U_store, _, _, _ = simulate()

            if U_store.std() < 20.0:  # arbitrary threshold
                torch.save(U_store, args.dir_path + f"/{args.idx}.pt")
                break

        except RuntimeWarning as e:
            print(f"A RuntimeWarning occurred: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, help="Path to save the data")
    parser.add_argument("--idx", type=int, help="Index of the data")
    args = parser.parse_args()
    main(args)
