import json, math, os, random, uuid
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "data/rheo_synth"
N_FIGS = 8
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

@dataclass
class SeriesTruth:
    label_in_legend: str
    Gp_plateau_Pa: float
    Gpp_plateau_Pa: float
    LVR_range_Pa: Tuple[float, float]
    yield_stress_Pa: float
    flow_point_Pa: float
    post_yield_slope_storage: float
    stress: List[float]
    Gp_curve: List[float]
    Gpp_curve: List[float]

def sample_params(rng: np.random.Generator, n_series_min=1, n_series_max=5):
    # stress grid sampling
    sigma_min = np.exp(rng.normal(np.log(20.0), 0.5))
    sigma_max = np.exp(rng.normal(np.log(8000.0), 0.4))
    
    if sigma_max < sigma_min * 30:
        sigma_max = sigma_min * 30

    n_decades = math.log10(sigma_max) - math.log10(sigma_min)
    pts_per_decade = rng.integers(6, 13)
    n_pts = int(max(30, min(160, pts_per_decade * max(1, int(round(n_decades))))))
    stress = np.logspace(np.log10(sigma_min), np.log10(sigma_max), n_pts)

    # number of series
    n_series = int(np.clip(np.random.poisson(3), n_series_min, n_series_max))
    assert n_series >= 1

    truths: List[SeriesTruth] = []
    label_pool = ["SiO2", "W", "WO3", "B", "B/Gd2O3", "Al2O3", "CaCO3"]
    rng.shuffle(label_pool)
    labels = label_pool[:n_series]

    for i in range(n_series):
        Gp_plateau = np.exp(rng.normal(np.log(1e5), 0.7))  # ~1e4-1e6 typical
        r = np.random.beta(2, 14)                          # G''/G' ratio at plateau
        Gpp_plateau = r * Gp_plateau

        gamma_c = np.exp(rng.normal(np.log(0.03), 0.6))    # 1-5% typical
        eta_y = np.exp(rng.normal(0.0, 0.25))
        tau_y = Gp_plateau * gamma_c * eta_y
        kappa = np.exp(rng.normal(np.log(2.5), 0.3))
        flow_point = tau_y * kappa
        slope_after = float(np.clip(rng.normal(-0.6, 0.2), -1.2, -0.2))

        # Slight LVR slopes
        lvr_gp_slope = rng.normal(0.0, 0.01)  # near flat in log-log
        lvr_gpp_slope = abs(rng.normal(0.03, 0.02))

        Gp = np.empty_like(stress)
        Gpp = np.empty_like(stress)

        # Construct curves
        for j, s in enumerate(stress):
            if s < tau_y:
                # LVR
                decades = (np.log10(s) - np.log10(stress[0]))
                Gp[j] = Gp_plateau * (10 ** (lvr_gp_slope * decades))
                Gpp[j] = Gpp_plateau * (10 ** (lvr_gpp_slope * decades))
            else:
                # Post-yield softening for G'
                Gp[j] = Gp_plateau * (s / tau_y) ** slope_after
                # Loss modulus bump around tau_y then taper
                width = np.exp(rng.normal(np.log(0.3 * tau_y), 0.4))
                bump = 1.0 + max(0.0, rng.normal(0.2, 0.08)) * np.exp(-0.5 * ((np.log(s) - np.log(tau_y)) / (np.log(1 + width / tau_y))) ** 2)
                Gpp[j] = Gpp_plateau * bump * (s / tau_y) ** max(0.0, -0.15)  # near-flat/tiny decay

        # Enforce crossover near flow point (blend toward equality at tau_f)
        idx_f = (np.abs(stress - flow_point)).argmin()
        alpha = 0.35
        Gp[idx_f:] = (1 - alpha) * Gp[idx_f:] + alpha * Gpp[idx_f:]
        Gpp[idx_f:] = (1 - alpha) * Gpp[idx_f:] + alpha * Gp[idx_f:]

        # Multiplicative noise
        noise_gp = np.exp(rng.normal(0, rng.uniform(0.03, 0.06), size=Gp.shape))
        noise_gpp = np.exp(rng.normal(0, rng.uniform(0.03, 0.06), size=Gpp.shape))
        Gp *= noise_gp
        Gpp *= noise_gpp

        # Clipping to plotting bounds
        Gp = np.clip(Gp, 10.0, 1e7)
        Gpp = np.clip(Gpp, 10.0, 1e7)

        # Determine LVR range as [min stress, min(tau_y, 0.7*flow_point)]
        lvr_hi = float(min(tau_y, 0.7 * flow_point))
        lvr_lo = float(stress[0])

        truths.append(
            SeriesTruth(
                label_in_legend=labels[i],
                Gp_plateau_Pa=float(Gp_plateau),
                Gpp_plateau_Pa=float(Gpp_plateau),
                LVR_range_Pa=(float(lvr_lo), float(lvr_hi)),
                yield_stress_Pa=float(tau_y),
                flow_point_Pa=float(flow_point),
                post_yield_slope_storage=float(slope_after),
                stress=stress.tolist(),
                Gp_curve=Gp.tolist(),
                Gpp_curve=Gpp.tolist(),
            )
        )

    assert len(truths) == n_series
    return stress, truths

def render_and_save(fig_id: str, stress: np.ndarray, truths: List[SeriesTruth], out_dir: str) -> Dict:
    assert len(truths) >= 1
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(7, 4))
    for t in truths:
        # Storage: squares; Loss: circles; allow default colors
        plt.plot(stress, t.Gp_curve, marker="s", linestyle="-", linewidth=1.2, markersize=3, label=f"{t.label_in_legend} (G')")
        plt.plot(stress, t.Gpp_curve, marker="o", linestyle="-", linewidth=1.0, markersize=3, label=f"{t.label_in_legend} (G'')")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Oscillation stress (Pa)")
    plt.ylabel("Modulus (Pa)")
    plt.title("Synthetic SAOS Stress Sweep")
    plt.legend(ncol=2, fontsize=8, handlelength=1.8)
    plt.tight_layout()

    img_path = os.path.join(out_dir, f"{fig_id}.png")
    plt.savefig(img_path, dpi=200)
    plt.close()

    # Build JSON annotation
    annotation = {
        "figure_id": fig_id,
        "axes": {"x_unit": "Pa", "y_unit": "Pa", "x_scale": "log10", "y_scale": "log10"},
        "materials": [
            {
                "label_in_legend": t.label_in_legend,
                "Gp_plateau_Pa": t.Gp_plateau_Pa,
                "Gpp_plateau_Pa": t.Gpp_plateau_Pa,
                "LVR_range_Pa": [t.LVR_range_Pa[0], t.LVR_range_Pa[1]],
                "yield_stress_Pa": t.yield_stress_Pa,
                "flow_point_Pa": t.flow_point_Pa,
                "post_yield_slope_storage": t.post_yield_slope_storage,
            }
            for t in truths
        ],
    }

    json_path = os.path.join(out_dir, f"{fig_id}.json")
    with open(json_path, "w") as f:
        json.dump(annotation, f, indent=2)

    return {"image": img_path, "annotation": json_path}

def main(n_figs: int = N_FIGS, out_dir: str = OUT_DIR, seed: int = SEED) -> List[Dict]:
    assert n_figs > 0
    rng = np.random.default_rng(seed)
    outputs = []
    for _ in range(n_figs):
        stress, truths = sample_params(rng)
        fig_id = str(uuid.uuid4())[:8]
        out = render_and_save(fig_id, stress, truths, out_dir)
        outputs.append(out)
    return outputs

outputs = main()
outputs
