import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import argparse



def load_stefan_entropy(path, has_header=True):
    """
    Load two-column CSV file: l, S(l)
    """
    if has_header:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
    else:
        data = np.loadtxt(path, delimiter=",")

    l = data[:, 0]
    S = data[:, 1]
    return l, S



def thermal_integral_z(zmin, zmax, eps_u=1e-8):
    """
    Compute I = ∫_{zmin}^{zmax} dz / ((1+z)^2 ln z)
    using substitution z = exp(u) to improve numerical stability.
    """
    if zmax <= zmin:
        return 0.0

    umin = np.log(zmin)
    umax = np.log(zmax)

    # integrand in u-variable
    def integrand_u(u):
        # avoid 1/u singularity exactly at u=0 (z=1)
        if abs(u) < eps_u:
            # principal value would be finite, but physically we never integrate exactly through z=1
            return 0.0
        eu = np.exp(u)
        return (eu / (1.0 + eu)**2) * (1.0 / u)

    val, _ = quad(integrand_u, umin, umax, limit=400)
    return val


def S_analytic_finite_L(l_vals, L, T, alpha, Omega0=1.0, z_floor=1.0 + 1e-6):
    """
    Implements Eq. (slL) exactly, with numerically stable integration.

    S_l(T) = (ln2/6) [ ln fL(l) - (2/alpha) * I(zmin,zmax) ]
    where I = ∫ dz / ((1+z)^2 ln z),
    zmax = Omega0/(2T), zmin = Omega0/(2T fL(l)^alpha).
    """
    S_th = []

    for l in l_vals:
        if l <= 0 or l >= L:
            S_th.append(0.0)
            continue

        fL = (L / np.pi) * np.sin(np.pi * l / L)

        # T=0 formula (Calabrese-Cardy form with ln2/6 prefactor)
        if T == 0:
            # Note: this can be negative for very small l because fL<1; that's a known finite-size/cutoff issue.
            S_th.append((np.log(2) / 6.0) * np.log(fL))
            continue

        z_max = Omega0 / (2.0 * T)
        z_min = Omega0 / (2.0 * T * (fL**alpha))

        # Numerical safety: avoid integrating across z=1 where ln z = 0
        # If bounds fall below 1, lift them slightly above 1.
        z_max = max(z_max, z_floor)
        z_min = max(z_min, z_floor)

        # If z_min > z_max, the integral is zero (no thermal correction in this regime)
        if z_min >= z_max:
            corr = 0.0
        else:
            corr = (2.0 / alpha) * thermal_integral_z(z_min, z_max)

        S_val = (np.log(2) / 6.0) * (np.log(fL) - corr)

        # Physical post-processing: entropy cannot be negative
        # This does NOT modify Eq.(slL); it enforces S>=0 for plotting/interpretation.
        S_th.append(max(S_val, 0.0))

    return np.array(S_th)


parser = argparse.ArgumentParser(description="Plot SDRG-X entanglement entropy")
parser.add_argument("data_folder", nargs="?", default="sdrgX_data_numba", help="Path to data folder")
args = parser.parse_args()
data_folder = args.data_folder

# Load data
with open(f"{data_folder}/S_l_all_T.json") as f:
    data = json.load(f)

L = data["L"]
alpha = data["alpha"]
l_vals = np.arange(0, L)

T_list = sorted(map(float, data["S_l_by_T"].keys()))

plt.figure(figsize=(6, 4))

for T in T_list[:-1]:  # skip the very highest T for clarity
    print(f"Processing T={T:g}...")
    S = np.array(data["S_l_by_T"][str(T)])

    # ---- numerical SDRG-X ----
    plt.plot(
        l_vals,
        S,
        label=rf"$T={T:g}$ (num)"
    )

    # ---- Stefan data ----
    stefan_path = f"data_stefan/S_l_T_{T:g}.csv"
    try:
        l_st, S_st = load_stefan_entropy(stefan_path)

        plt.plot(
            l_st,
            S_st,
            "--",
            linewidth=1.5,
            label=rf"$T={T:g}$ (theory)"
        )
    except OSError:
        print(f"Warning: Stefan data not found for T={T:g}")

    # ---- analytical curve ----
    # S_th = S_analytic_finite_L(
    #     l_vals=l_vals,
    #     L=L,
    #     T=T,
    #     alpha=data["alpha"],
    #     Omega0=1.0
    # )


    # plt.plot(
    #     l_vals,
    #     S_th+0.0,  # slight vertical offset for visibility
    #     "--",
    #     linewidth=1.5,
    #     label=rf"$T={T:g}$ (theory)"
    # )

plt.xlabel(r"$\ell$")
plt.ylabel(r"$S(\ell)$")
plt.title(
    rf"SDRG-X entanglement entropy "
    rf"($N={data['N']},\,\alpha={alpha}$)"
)
plt.legend(fontsize=8, ncol=2)
#plt.grid(True)
plt.tight_layout()

outname = (
    f"EE_SDRG_X_"
    f"N{data['N']}_"
    f"alpha{alpha}_"
    f"Tmulti_with_theory.png"
)
plt.savefig(outname, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure as {outname}")

