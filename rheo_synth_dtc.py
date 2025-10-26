# Synthetic SAOS generator (sigmoid-blended), DIW-style
# - Enforces G'(tau_min) > G''(tau_min)
# - Enforces exact G'==G'' at flow point (tau_f) on-grid
# - Allows controlling LVR ratio r_eff_target = G''/G' at the low-stress bound
# - Pins stress grid so tau_f is exactly a tick
#
# Usage:
#   python rheo_sigmoid_generator.py --out /tmp/rheo --n 4 --seed 123 --r_eff 0.1
#
# Notes:
#   * Ground-truth parameters emitted in JSON.
#   * Stress grid is constructed around tau_f to include it exactly.

import os, json, math, uuid, argparse, random
import string
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Series:
    label_in_legend: str
    params: Dict[str, float]
    stress: List[float]
    Gp_curve: List[float]
    Gpp_curve: List[float]

def sample_and_generate(rng: np.random.Generator,
                        r_eff_target: float = 0.10,
                        pts_per_decade: int = 10,
                        low_decades: float = 1.5,
                        high_decades: float = 1.0,
                        n_series_min: int = 1,
                        n_series_max: int = 4
                       ): 
    
    # ----- Sample interpretable parameters shared by grid construction -----
    # Pick plateau modulus and yield/flow stresses

    # Gp = float(np.exp(rng.normal(np.log(1e5), np.log(1e1))))         # Pa
    # gamma_c = float(np.exp(rng.normal(np.log(0.03), 0.5)))    # critical strain ~ few %
    # tau_y = float(Gp * gamma_c * np.exp(rng.normal(0.0, 0.20)))
    # kappa = float(np.exp(rng.normal(np.log(1), 0.20)))
    # tau_f = float(kappa * tau_y)

    # assert tau_f > tau_y > 0 and Gp > 0

    # ----- Build stress grid with tau_f pinned as a tick -----
    assert pts_per_decade >= 6 and low_decades > 0 and high_decades > 0

    dlog = 1.0 / float(pts_per_decade)                # log10 spacing per point
    n_low = int(math.ceil(low_decades * pts_per_decade))
    n_high = int(math.ceil(high_decades * pts_per_decade))
    exps = np.arange(-n_low, n_high + 1, dtype=float) * dlog
    # stress = tau_f * (10.0 ** exps)

    # Sanity bands (typical DIW): allow 5–10^5 Pa range but not strictly enforced
    # assert stress[0] < tau_y and stress[-1] > tau_f

    # ----- Series count and labels -----
    n_series = int(np.clip(np.random.poisson(3), n_series_min, n_series_max))
    assert n_series >= 1

    # return list of len(n_series) with elements lables of random characters
    characters = list(string.ascii_letters)

    labels = []
    for i in range(n_series): 
        label = np.random.choice(characters, size=np.random.randint(low=3, high=10))
        labels.append(''.join(label))

    all_series = []
    for i in range(n_series):

        Gp = float(np.exp(np.random.normal(np.log(1e5), 3)))         # Pa
        gamma_c = float(np.exp(np.random.normal(np.log(0.03), 0.25)))    # critical strain ~ few %
        tau_y = float(Gp * gamma_c * np.exp(np.random.normal(0.0, 0.20)))
        kappa = float(np.exp(rng.normal(np.log(2), 0.20)))
        tau_f = float(kappa * tau_y)

        stress = (tau_f / 10) * (10.0 ** exps)

        print(f'flow stress: {tau_f}')
        print(f'yield stress: {tau_y}')
        print(f'LVR G prime {Gp}')

        assert tau_f > tau_y > 0 and Gp > 0

        assert stress[0] < tau_y and stress[-1] > tau_f

        # Series-specific parameters
        # Storage modulus drop (sharpening) and loss tail
        m = float(rng.uniform(1.9, 2.0))
        n = float(rng.uniform(1.9, 2.0))
        alpha = float(rng.uniform(0.85, 1.15))
        beta  = float(rng.uniform(0.85, 1.15))
        A = float(rng.uniform(0.12, 0.30))
        w = float(rng.uniform(0.35, 0.65))

        def Gp_func(t):
            return Gp / (1.0 + (t / tau_y) ** m) ** alpha

        def bump_base(t):
            # modest peak near tau_y
            return 1.0 + A * np.exp(-((np.log(t / tau_y)) ** 2) / (2 * w * w))

        def Gpp_base(t):
            # base (unscaled) loss without LVR-ratio or flow equality constraints
            return bump_base(t) / (1.0 + (t / tau_f) ** n) ** beta

        # --- Two-constraint scaling for G'': (i) exact equality at tau_f, (ii) r_eff at low end ---
        idx_f = int(np.argmin(np.abs(stress - tau_f)))
        t_f = float(stress[idx_f])
        Gp_tf = float(Gp_func(t_f))
        B_tf = float(Gpp_base(t_f))         # base shape value at tau_f
        s = Gp_tf / max(B_tf, 1e-18)        # primary scale so G''(tau_f)=G'(tau_f)

        t_min = float(stress[0])
        Gp_min = float(Gp_func(t_min))
        B_min = float(Gpp_base(t_min))

        # Smooth shape adjustment f(t) = (t/t_f)^q so that f(t_f)=1 and
        # r_eff_target = G''(t_min)/G'(t_min) is met exactly.
        desired = r_eff_target * Gp_min / max(s * B_min, 1e-18)
        # Guard: t_min < t_f, desired>0
        assert t_min < t_f and desired > 0
        q = math.log(desired) / math.log(t_min / t_f)

        def shape_adj(t):
            return (t / t_f) ** q  # equals 1 at t_f; adjusts low-end ratio

        def Gpp_func(t):
            return s * shape_adj(t) * Gpp_base(t)

        # --- Build curves + small multiplicative noise ---
        noise_gp = np.exp(rng.normal(0, rng.uniform(0.02, 0.05), size=stress.shape))
        noise_gpp = np.exp(rng.normal(0, rng.uniform(0.02, 0.05), size=stress.shape))
        Gp_curve = Gp_func(stress) * noise_gp
        Gpp_curve = Gpp_func(stress) * noise_gpp

        # Enforce exact equality at idx_f after noise by snapping both to their mean
        eq_val = 0.5 * (Gp_curve[idx_f] + Gpp_curve[idx_f])
        Gp_curve[idx_f] = eq_val
        Gpp_curve[idx_f] = eq_val

        # Sanity: LVR inequality at the minimum stress and exact flow equality
        assert Gp_curve[0] > Gpp_curve[0]
        assert abs(Gp_curve[idx_f] - Gpp_curve[idx_f]) / max(Gp_curve[idx_f], 1e-12) < 1e-9

        params = {
            "Gp_plateau_Pa": Gp,
            "tau_y_Pa": tau_y,
            "tau_f_Pa": t_f,
            "r_eff_LVR": float(Gpp_curve[0] / max(Gp_curve[0], 1e-12)),
            "m": m, "n": n, "alpha": alpha, "beta": beta, "A": A, "w": w,
            "q_shape": q
        }

        all_series.append(Series(
            label_in_legend=labels[i],
            params=params,
            stress=stress.tolist(),
            Gp_curve=Gp_curve.tolist(),
            Gpp_curve=Gpp_curve.tolist(),
        ))

    return stress, all_series

def render_and_save(fig_id, stress, series, out_dir):
    assert len(series) >= 1
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(7.2, 4.2))
    for s in series:
        plt.plot(stress, s.Gp_curve, marker="s", linestyle="-", linewidth=1.2, markersize=3, label=f"{s.label_in_legend} (G')")
        plt.plot(stress, s.Gpp_curve, marker="o", linestyle="-", linewidth=1.0, markersize=3, label=f"{s.label_in_legend} (G'')")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Oscillation stress (Pa)"); plt.ylabel("Modulus (Pa)")
    plt.title("Synthetic SAOS (Sigmoid-Blended, constrained)")
    plt.legend(ncol=2, fontsize=8, handlelength=1.8)
    plt.tight_layout()

    img_path = os.path.join(out_dir, f"{fig_id}.png")
    plt.savefig(img_path, dpi=220); plt.close()

    ann = {
        "figure_id": fig_id,
        "axes": {"x_unit": "Pa", "y_unit": "Pa", "x_scale": "log10", "y_scale": "log10"},
        "materials": [
            {"label_in_legend": s.label_in_legend, **s.params} for s in series
        ]
    }
    json_path = os.path.join(out_dir, f"{fig_id}.json")
    with open(json_path, "w") as f:
        json.dump(ann, f, indent=2)

    return {"image": img_path, "annotation": json_path}

def main(n_figs: int = 6,
         out_dir: str = "data/rheo_sigmoid",
         seed: int = 123,
         r_eff_target: float = 0.10,
         pts_per_decade: int = 10,
         low_decades: float = 1.5,
         high_decades: float = 2.0
        ):
    
    assert n_figs > 0 and pts_per_decade >= 6

    rng = np.random.default_rng(seed)

    outputs = []

    for _ in range(n_figs):
        stress, series = sample_and_generate(rng,
                                             r_eff_target=r_eff_target,
                                             pts_per_decade=pts_per_decade,
                                             low_decades=low_decades,
                                             high_decades=high_decades)
        fig_id = str(uuid.uuid4())[:8]
        outputs.append(render_and_save(fig_id, stress, series, out_dir))
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/rheo_sigmoid", help="Output directory")
    parser.add_argument("--n", type=int, default=6, help="Number of figures")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed")
    parser.add_argument("--r_eff", type=float, default=0.10, help="Target G''/G' ratio at low-stress bound (LVR)")
    parser.add_argument("--ppd", type=int, default=10, help="Points per decade (>=6)")
    parser.add_argument("--low_dec", type=float, default=1.5, help="Decades below tau_f")
    parser.add_argument("--high_dec", type=float, default=2.0, help="Decades above tau_f")
    args = parser.parse_args()

    outputs = main(n_figs=args.n,
                   out_dir=args.out,
                   seed=args.seed,
                   r_eff_target=args.r_eff,
                   pts_per_decade=args.ppd,
                   low_decades=args.low_dec,
                   high_decades=args.high_dec)

    print(json.dumps(outputs, indent=2))
