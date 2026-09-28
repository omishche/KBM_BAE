#!/usr/bin/env python3
"""
kbm_bae_streamlit.py
=====================

Streamlit web application for the kinetic BAE/KBM dispersion relation
of Zonca, Chen & Santoro, Plasma Phys. Control. Fusion 38, 2011 (1996).

Run with:
    streamlit run kbm_bae_streamlit.py
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

try:
    import tables  # PyTables, optional (for HDF5 export)
    HAS_PYTABLES = True
except ImportError:
    HAS_PYTABLES = False


# =============================================================================
# 1. Core physics: Eq. (24) and root finding
# =============================================================================

def eq24(Omega_vec, tau, q, eta_i, Omega_star_ni, Omega_star_Ti, Omega_star_pi):
    """Real/imaginary parts of Eq. (24) with Lambda = 0."""
    Omega = Omega_vec[0] + 1j * Omega_vec[1]

    term1 = (Omega**2 - 7.0/4.0 * q**2) * (1.0 - Omega_star_pi / Omega)
    term2 = 7.0/4.0 * q**2 * Omega_star_Ti / Omega
    term3 = - q**2 * (1.0 - Omega_star_pi / Omega)**2 \
            / (1.0/tau + Omega_star_ni / Omega)

    bracket1 = Omega - Omega_star_ni - eta_i * Omega**2 * Omega_star_ni
    bracket2 = (Omega**2
                + (1.0 - Omega_star_pi / Omega)
                / (1.0/tau + Omega_star_ni / Omega))**2

    term4 = (1j * np.sqrt(np.pi) * q**2
             * np.exp(-Omega**2) * bracket1 * bracket2)

    F = term1 + term2 + term3 + term4
    return [F.real, F.imag]


def solve_single(Omega_init, tau, q, eta_i, Omega_star_ni, Omega_star_Ti, Omega_star_pi):
    """Newton solve for a single root, starting from a complex guess."""
    sol = fsolve(eq24, [Omega_init.real, Omega_init.imag],
                 args=(tau, q, eta_i, Omega_star_ni, Omega_star_Ti, Omega_star_pi),
                 full_output=True)
    x, info, ier, _ = sol
    residual = float(np.max(np.abs(info['fvec'])))
    return x[0] + 1j * x[1], residual, ier


def find_all_roots(tau, q, eta_i, Omega_star_ni, Omega_star_Ti, Omega_star_pi,
                   re_range=(-3.0, 3.0), im_range=(-1.0, 1.0),
                   n_re=25, n_im=25, tol=1e-8, unique_tol=1e-5):
    """Grid-search all unique roots of Eq. (24) in a complex-plane box."""
    re_vals = np.linspace(re_range[0], re_range[1], n_re)
    im_vals = np.linspace(im_range[0], im_range[1], n_im)
    roots = []
    for re0 in re_vals:
        for im0 in im_vals:
            if re0 == 0.0 and im0 == 0.0:
                continue
            root, residual, ier = solve_single(
                re0 + 1j * im0, tau, q, eta_i,
                Omega_star_ni, Omega_star_Ti, Omega_star_pi)
            if ier == 1 and residual < tol:
                if all(abs(root - r) > unique_tol for r in roots):
                    roots.append(root)
    return sorted(roots, key=lambda z: z.real)


# =============================================================================
# 2. Analytic seeds from Eqs. (25)-(26)
# =============================================================================

def eq26_Omega0(q, tau, Omega_star_ni):
    """
    Omega_0^2 = (7/4+tau) q^2   for Omega_*ni << |Omega_0|
    Omega_0^2 = (3/4) q^2       for Omega_*ni >> |Omega_0|
    """
    candidates = [(7.0/4.0 + tau) * q**2, (3.0/4.0) * q**2]
    Omega0 = None
    for idx, Omega0_sq in enumerate(candidates):
        cand = np.sqrt(Omega0_sq)
        if Omega_star_ni < cand and idx == 0:
            Omega0 = cand
            break
        if Omega_star_ni >= cand and idx == 1:
            Omega0 = cand
            break
    if Omega0 is None:
        Omega0 = np.sqrt(max(candidates))
    return Omega0


def eq25_26_seed(q, tau, Omega_star_ni, branch):
    """Accumulation-point seed at eta_i -> 0 (Eqs. 25-26)."""
    if branch == "KBM":
        return complex(Omega_star_ni, 0.0)
    elif branch == "BAE":
        Omega0 = eq26_Omega0(q, tau, Omega_star_ni)
        return (Omega0 - 1j * np.sqrt(np.pi) / 2.0
                * q**2 * Omega0**4 * np.exp(-Omega0**2))
    else:
        raise ValueError(f"branch='{branch}' not recognized.")


def omega_star_at(scan_type, scan_val, params):
    """Map scan parameter (eta_i or ky) to (Omega_*ni, Omega_*Ti, Omega_*pi, eta_i)."""
    if scan_type == "eta_i":
        eta_local = scan_val
        Oni = params["Omega_star_ni"]
    elif scan_type == "ky":
        eta_local = params["eta_i"]
        Oni = params["Cn"] * scan_val
    else:
        raise ValueError(scan_type)
    OTi = eta_local * Oni
    Opi = Oni + OTi
    return Oni, OTi, Opi, eta_local


def track_branch(scan_vals, scan_type, params, branch, tol=1e-8):
    """Continuation tracking of a single branch across the scan."""
    Oni0, _, _, _ = omega_star_at(scan_type, scan_vals[0], params)
    guess = eq25_26_seed(params["q"], params["tau"], Oni0, branch)

    result = np.full(len(scan_vals), np.nan + 1j*np.nan, dtype=complex)
    for k, val in enumerate(scan_vals):
        Oni, OTi, Opi, eta_local = omega_star_at(scan_type, val, params)
        root, residual, ier = solve_single(
            guess, params["tau"], params["q"], eta_local, Oni, OTi, Opi)
        if ier == 1 and residual < tol:
            result[k] = root
            guess = root
        else:
            result[k] = np.nan + 1j*np.nan
    return result


def reference_curves(scan_vals, scan_type, params):
    """Real-part reference curves: Omega_*pi (Eq. 25) and Omega_0 (Eq. 26)."""
    ref_KBM = np.empty(len(scan_vals))
    ref_BAE = np.empty(len(scan_vals))
    for k, val in enumerate(scan_vals):
        Oni, OTi, Opi, _ = omega_star_at(scan_type, val, params)
        ref_KBM[k] = Opi
        ref_BAE[k] = eq26_Omega0(params["q"], params["tau"], Oni)
    return ref_KBM, ref_BAE


# =============================================================================
# 3. Plotting helpers
# =============================================================================

def make_scan_figure(scan_vals, scan_label, branches, references, title=""):
    """Create a two-panel (Re, Im) Matplotlib figure."""
    fig, (ax_re, ax_im) = plt.subplots(1, 2, figsize=(12, 4.8))

    colors = {"KBM": "tab:blue", "BAE": "tab:red"}

    for label, arr in branches.items():
        c = colors.get(label)
        ax_re.plot(scan_vals, arr.real, '-o', color=c, markersize=3,
                  linewidth=1.5, label=f"{label} (tracked)")
        ax_im.plot(scan_vals, arr.imag, '-o', color=c, markersize=3,
                  linewidth=1.5, label=f"{label} (tracked)")

    if references:
        for label, arr in references.items():
            ax_re.plot(scan_vals, arr, '--', linewidth=2, label=label)

    ax_im.axhline(0, color='k', linewidth=0.7, linestyle='--')

    ax_re.set_xlabel(scan_label)
    ax_re.set_ylabel(r'Re($\Omega$)')
    ax_re.set_title(r'Re($\Omega$)')
    ax_re.legend(fontsize=8)
    ax_re.grid(alpha=0.3)

    ax_im.set_xlabel(scan_label)
    ax_im.set_ylabel(r'Im($\Omega$)')
    ax_im.set_title(r'Im($\Omega$)  (growth rate)')
    ax_im.legend(fontsize=8)
    ax_im.grid(alpha=0.3)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def make_single_point_figure(roots, title=""):
    """Create a complex-plane scatter plot of roots."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in roots:
        ax.plot(r.real, r.imag, 'ro', markersize=8)
        ax.annotate(f"  ({r.real:.3f}, {r.imag:.3f})",
                    (r.real, r.imag), fontsize=8)
    ax.axhline(0, color='k', linewidth=0.6, linestyle='--')
    ax.axvline(0, color='k', linewidth=0.6, linestyle='--')
    ax.set_xlabel(r'Re($\Omega$)')
    ax.set_ylabel(r'Im($\Omega$)')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# =============================================================================
# 4. HDF5 export helper
# =============================================================================

def export_to_hdf5_bytes(scan_vals, scan_type, branches):
    """
    Build an in-memory HDF5 file (via PyTables) containing the scan
    results, and return it as raw bytes suitable for st.download_button.
    """
    n = len(scan_vals)

    class ResultRow(tables.IsDescription):
        scan_val = tables.Float64Col()
        Re_KBM   = tables.Float64Col()
        Im_KBM   = tables.Float64Col()
        Re_BAE   = tables.Float64Col()
        Im_BAE   = tables.Float64Col()

    buf = io.BytesIO()
    # PyTables needs a real filename with the driver "H5FD_CORE" to work
    # fully in-memory; we use driver_core_backing_store=0 to avoid disk I/O.
    h5f = tables.open_file(
        "in_memory.h5", mode="w", title="KBM/BAE scan results",
        driver="H5FD_CORE", driver_core_backing_store=0
    )
    try:
        group = h5f.create_group("/", "scan", "Scan results")
        table = h5f.create_table(group, "results", ResultRow,
                                 "Omega(scan_val) for KBM/BAE branches")
        row = table.row

        KBM_arr = branches.get("KBM", np.full(n, np.nan + 1j*np.nan))
        BAE_arr = branches.get("BAE", np.full(n, np.nan + 1j*np.nan))

        for i in range(n):
            row["scan_val"] = scan_vals[i]
            row["Re_KBM"] = KBM_arr[i].real
            row["Im_KBM"] = KBM_arr[i].imag
            row["Re_BAE"] = BAE_arr[i].real
            row["Im_BAE"] = BAE_arr[i].imag
            row.append()
        table.flush()
        table.attrs.scan_type = scan_type

        # Extract raw bytes from the in-core HDF5 image
        raw_bytes = h5f.get_file_image()
    finally:
        h5f.close()

    return raw_bytes


# =============================================================================
# 5. Streamlit app
# =============================================================================

def main():
    st.set_page_config(page_title="KBM/BAE Dispersion Solver", layout="wide")

    st.title("KBM / BAE Dispersion Relation Solver")
    st.caption("Zonca, Chen & Santoro, *Plasma Phys. Control. Fusion* "
              "**38**, 2011 (1996) — Eq. (24)")

    # ------------------------------------------------------------------
    # Sidebar: inputs
    # ------------------------------------------------------------------
    st.sidebar.header("Scan configuration")

    scan_type_label = st.sidebar.selectbox(
        "Scan over:", ["eta_i", "ky", "single point"])

    st.sidebar.header("Fixed equilibrium parameters")
    tau = st.sidebar.number_input("tau (T_e / T_i)", value=1.0, step=0.1,
                                  format="%.4f")
    q = st.sidebar.number_input("q (safety factor)", value=1.5, step=0.1,
                                format="%.4f")

    params = {"tau": tau, "q": q}

    branches_wanted = []
    references_wanted = True
    n_scan = 80

    if scan_type_label == "eta_i":
        st.sidebar.header("eta_i scan parameters")
        Omega_star_ni = st.sidebar.number_input(
            "Omega_*ni (fixed)", value=1.0, step=0.1, format="%.4f")
        eta_min = st.sidebar.number_input(
            "eta_i min", value=0.0001, step=0.01, format="%.4f")
        eta_max = st.sidebar.number_input(
            "eta_i max", value=2.0, step=0.1, format="%.4f")
        n_scan = st.sidebar.slider("Number of scan points", 10, 300, 80)

        params["Omega_star_ni"] = Omega_star_ni
        scan_vals = np.linspace(eta_min, eta_max, n_scan)
        scan_label = r"$\eta_i$"
        scan_key = "eta_i"

    elif scan_type_label == "ky":
        st.sidebar.header("ky scan parameters")
        eta_i_fixed = st.sidebar.number_input(
            "eta_i (fixed)", value=0.5, step=0.1, format="%.4f")
        Cn = st.sidebar.number_input(
            "Cn  (Omega_*ni = Cn * ky)", value=1.0, step=0.1, format="%.4f")
        ky_min = st.sidebar.number_input(
            "ky min", value=0.05, step=0.05, format="%.4f")
        ky_max = st.sidebar.number_input(
            "ky max", value=3.0, step=0.1, format="%.4f")
        n_scan = st.sidebar.slider("Number of scan points", 10, 300, 100)

        params["eta_i"] = eta_i_fixed
        params["Cn"] = Cn
        scan_vals = np.linspace(ky_min, ky_max, n_scan)
        scan_label = r"$k_y$"
        scan_key = "ky"

    else:  # single point
        st.sidebar.header("Single-point search box")
        eta_i_sp = st.sidebar.number_input(
            "eta_i", value=0.5, step=0.1, format="%.4f")
        Omega_star_ni_sp = st.sidebar.number_input(
            "Omega_*ni", value=1.0, step=0.1, format="%.4f")
        re_min = st.sidebar.number_input("Re(Omega) min", value=-3.0)
        re_max = st.sidebar.number_input("Re(Omega) max", value=3.0)
        im_min = st.sidebar.number_input("Im(Omega) min", value=-1.0)
        im_max = st.sidebar.number_input("Im(Omega) max", value=1.0)
        grid_n = st.sidebar.slider("Search grid resolution", 10, 50, 25)

    if scan_type_label in ("eta_i", "ky"):
        st.sidebar.header("Branches")
        show_kbm = st.sidebar.checkbox("Track KBM", value=True)
        show_bae = st.sidebar.checkbox("Track BAE", value=True)
        references_wanted = st.sidebar.checkbox(
            "Show Eq.(25)-(26) reference curves", value=True)
        if show_kbm:
            branches_wanted.append("KBM")
        if show_bae:
            branches_wanted.append("BAE")

    run_button = st.sidebar.button("Run", type="primary")

    # ------------------------------------------------------------------
    # Main area
    # ------------------------------------------------------------------
    if run_button:
        if scan_type_label == "single point":
            _run_single_point(tau, q, eta_i_sp, Omega_star_ni_sp,
                              re_min, re_max, im_min, im_max, grid_n)
        else:
            if not branches_wanted:
                st.error("Please select at least one branch (KBM and/or BAE).")
                return
            _run_scan(scan_vals, scan_key, scan_label, params,
                     branches_wanted, references_wanted)
    else:
        st.info("Set parameters in the sidebar and click **Run**.")
        with st.expander("About this app"):
            st.markdown(r"""
            This app solves the kinetic dispersion relation for
            low-frequency Alfvén modes (BAE and KBM branches) derived in

            > F. Zonca, L. Chen and R. A. Santoro,
            > *Kinetic theory of low-frequency Alfvén modes in tokamaks*,
            > Plasma Phys. Control. Fusion **38**, 2011 (1996).

            **Eq. (24)** (with $\Lambda=0$) is solved numerically for the
            complex normalized frequency $\Omega=\omega/\omega_{ti}$.

            - The **KBM** and **BAE** branches are tracked by numerical
              continuation, starting from the analytic accumulation
              points of **Eqs. (25)-(26)** at $\eta_i \to 0$.
            - You can scan over the ion temperature-gradient parameter
              $\eta_i$ (at fixed $\Omega_{*ni}$), or over the
              normalized poloidal wavenumber $k_y$ (at fixed $\eta_i$,
              assuming $\Omega_{*ni}(k_y) = C_n k_y$).
            - A **single-point** mode performs a grid search for all
              roots of Eq. (24) in a user-specified box of the complex
              $\Omega$-plane.
            """)


def _run_scan(scan_vals, scan_key, scan_label, params, branches_wanted,
             references_wanted):
    with st.spinner("Tracking branches..."):
        branches = {}
        for b in branches_wanted:
            branches[b] = track_branch(scan_vals, scan_key, params, b)

        references = None
        if references_wanted:
            ref_KBM, ref_BAE = reference_curves(scan_vals, scan_key, params)
            references = {
                r"$\Omega_{*pi}$ (Eq. 25)": ref_KBM,
                r"$\Omega_0$ (Eq. 26)": ref_BAE,
            }

    title_parts = [f"tau={params['tau']}", f"q={params['q']}"]
    if scan_key == "eta_i":
        title_parts.append(f"Omega_*ni={params['Omega_star_ni']}")
    else:
        title_parts.append(f"eta_i={params['eta_i']}")
        title_parts.append(f"Cn={params['Cn']}")
    title = ", ".join(title_parts)

    fig = make_scan_figure(scan_vals, scan_label, branches, references,
                           title=title)
    st.pyplot(fig)

    # --- Results table ---
    st.subheader("Results table")
    data = {scan_key: scan_vals}
    for label, arr in branches.items():
        data[f"Re({label})"] = arr.real
        data[f"Im({label})"] = arr.imag
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    # --- Download buttons ---
    col1, col2 = st.columns(2)

    with col1:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download as CSV",
            data=csv_bytes,
            file_name="kbm_bae_scan.csv",
            mime="text/csv",
        )

    with col2:
        if HAS_PYTABLES:
            try:
                h5_bytes = export_to_hdf5_bytes(scan_vals, scan_key, branches)
                st.download_button(
                    label="Download as HDF5 (PyTables)",
                    data=h5_bytes,
                    file_name="kbm_bae_scan.h5",
                    mime="application/x-hdf5",
                )
            except Exception as e:
                st.warning(f"HDF5 export failed: {e}")
        else:
            st.warning("Install `tables` (PyTables) to enable HDF5 export: "
                      "`pip install tables`")

    # --- Summary ---
    st.subheader("Summary (first / last scan point)")
    summary_rows = []
    for label, arr in branches.items():
        summary_rows.append({
            "Branch": label,
            f"Re at {scan_key}={scan_vals[0]:.4g}": arr[0].real,
            f"Im at {scan_key}={scan_vals[0]:.4g}": arr[0].imag,
            f"Re at {scan_key}={scan_vals[-1]:.4g}": arr[-1].real,
            f"Im at {scan_key}={scan_vals[-1]:.4g}": arr[-1].imag,
        })
    st.table(pd.DataFrame(summary_rows).set_index("Branch"))


def _run_single_point(tau, q, eta_i, Omega_star_ni,
                      re_min, re_max, im_min, im_max, grid_n):
    Omega_star_Ti = eta_i * Omega_star_ni
    Omega_star_pi = Omega_star_ni + Omega_star_Ti

    st.write(f"**Parameters:** tau={tau}, q={q}, eta_i={eta_i}, "
            f"Omega_*ni={Omega_star_ni}  "
            f"(Omega_*Ti={Omega_star_Ti:.4f}, Omega_*pi={Omega_star_pi:.4f})")

    with st.spinner("Searching for roots..."):
        roots = find_all_roots(
            tau, q, eta_i, Omega_star_ni, Omega_star_Ti, Omega_star_pi,
            re_range=(re_min, re_max), im_range=(im_min, im_max),
            n_re=grid_n, n_im=grid_n)

    if not roots:
        st.warning("No roots found in the specified search box. "
                  "Try widening the Re/Im ranges.")
        return

    st.success(f"Found {len(roots)} root(s).")

    fig = make_single_point_figure(
        roots, title=f"Roots of Eq.(24): tau={tau}, q={q}, eta_i={eta_i}")
    st.pyplot(fig)

    df = pd.DataFrame({
        "Re(Omega)": [r.real for r in roots],
        "Im(Omega)": [r.imag for r in roots],
    })
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download roots as CSV",
        data=csv_bytes,
        file_name="kbm_bae_roots.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
