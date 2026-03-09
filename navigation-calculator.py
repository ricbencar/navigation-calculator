"""
PIANC Under Keel Clearance (UKC) Calculator
==================================================
This program evaluates net under keel clearance by combining:

1. hydrodynamic squat from six empirical and semi-empirical methods;
2. a density-related draft increment;
3. heel-induced bilge penetration from wind and turning effects; and
4. wave-induced vertical motion allowance from three simplified methods.

The final UKC is obtained by subtracting the adopted dynamic allowances from
nominal water depth relative to static draft. Squat is taken as the average
of the three highest method outputs, and wave allowance is taken as the average
of the three implemented wave methods. Output is formatted as a fixed-width
ASCII engineering report suitable for console and GUI use.
"""

import math
import os
import textwrap
import function  # External wavelength model used by the wave-motion block.

# =============================================================================
# NUMERICAL UTILITIES
# =============================================================================

def get_input(prompt, default_val):
    """Safely retrieves user input, falling back to a default if left blank."""
    try:
        user_in = input(f"{prompt} [{default_val}]: ")
        return user_in if user_in.strip() else default_val
    except EOFError:
        return default_val

def interp_1d(x, xs, ys):
    """Standard 1D linear interpolation used for ROM 3.1 wave coefficients."""
    if x <= xs[0]: return ys[0]
    if x >= xs[-1]: return ys[-1]
    for i in range(len(xs)-1):
        if xs[i] <= x <= xs[i+1]:
            t = (x - xs[i]) / (xs[i+1] - xs[i])
            return ys[i] + t * (ys[i+1] - ys[i])
    return ys[-1]

# =============================================================================
# REPORT FORMATTING (FIXED-WIDTH ASCII BOX TABLES)
# =============================================================================

REPORT_WIDTH = 100  # total characters, including the two outer border characters


def _ellipsis(text: str, width: int) -> str:
    """Truncate *text* to *width* characters, appending an ellipsis if needed."""
    s = str(text)
    if width <= 0:
        return ""
    if len(s) <= width:
        return s
    if width <= 1:
        return s[:width]
    return s[: max(0, width - 1)] + "…"


def _pad(text: str, width: int, align: str) -> str:
    """Pad (and truncate) text to exactly *width* characters."""
    s = _ellipsis(text, width)
    if align == "right":
        return s.rjust(width)
    if align == "center":
        return s.center(width)
    return s.ljust(width)


def row_border(char: str = "=") -> str:
    """Return a full-width horizontal border line.

    The *char* argument is the border fill character.
    """
    fill = char if (isinstance(char, str) and len(char) == 1) else "="
    if fill == "+":
        fill = "="
    return "+" + (fill * (REPORT_WIDTH - 2)) + "+"


def row_title(title: str) -> str:
    """Return a centered title row."""
    inner = REPORT_WIDTH - 4
    return "| " + _pad(title, inner, "center") + " |"


def row_1col(text: str) -> str:
    """Return a single-span content row."""
    inner = REPORT_WIDTH - 4
    return "| " + _pad(text, inner, "left") + " |"


def row_1col_wrap(text: str, indent: int = 0):
    """Wrap *text* to the internal width of row_1col() and return framed rows.

    This prevents ellipsis truncation for long diagnosis/recommendation strings
    while preserving the fixed-width ASCII frame.
    """
    inner_width = REPORT_WIDTH - 4
    prefix = " " * max(0, int(indent))
    lines = textwrap.wrap(
        str(text),
        width=inner_width,
        subsequent_indent=prefix,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not lines:
        lines = [""]
    return [row_1col(line) for line in lines]


def row_3col(p1: str, p2: str, p3: str) -> str:
    """Return a 3-column row: Label | Symbol | Value."""
    # Total layout: "| " + c1 + " | " + c2 + " | " + c3 + " |"
    # Overhead = 10 chars; therefore c1+c2+c3 must equal REPORT_WIDTH-10.
    c1_w, c2_w = 40, 12
    c3_w = (REPORT_WIDTH - 10) - (c1_w + c2_w)
    return (
        "| "
        + _pad(p1, c1_w, "left")
        + " | "
        + _pad(p2, c2_w, "center")
        + " | "
        + _pad(p3, c3_w, "right")
        + " |"
    )


def row_2col(label: str, value: str) -> str:
    """Return a 2-column key/value row."""
    # Total layout: "| " + k + " : " + v + " |"
    # Overhead = 7 chars; therefore k+v must equal REPORT_WIDTH-7.
    k_w = 62
    v_w = (REPORT_WIDTH - 7) - k_w
    return "| " + _pad(label, k_w, "left") + " : " + _pad(value, v_w, "right") + " |"


# =============================================================================
# PIANC EMPIRICAL DATA TABLES

# Dictionary: [C_B_avg, GM/T_ratio, F_K, A_VF_a, A_VF_b, A_VL_a, A_VL_b]
# -----------------------------------------------------------------------------
# C_B_avg    : Average Block Coefficient for the vessel type.
# GM/T_ratio : Metacentric height to draft ratio (Stability approximation).
# F_K        : Bilge keel factor for dynamic heel penalties.
# A_VF / A_VL: Regression coefficients for Frontal and Lateral Windage Areas.
# =============================================================================
VESSEL_DATA = {
    'cargo':     [0.72, 0.20, 0.83,  0.592, 0.666,  3.213, 0.616],
    'bulk':      [0.85, 0.35, 0.88,  8.787, 0.370, 16.518, 0.425],
    'container': [0.60, 0.10, 0.78,  1.369, 0.609,  2.614, 0.703],
    'tanker':    [0.85, 0.35, 0.88,  2.946, 0.474,  3.598, 0.558],
    'roro':      [0.65, 0.15, 0.80, 10.697, 0.435, 28.411, 0.464],
    'passenger': [0.60, 0.15, 0.80,  8.842, 0.426,  3.888, 0.680],
    'ferry':     [0.60, 0.15, 0.80,  5.340, 0.473,  3.666, 0.674],
    'gas':       [0.75, 0.30, 0.83,  2.649, 0.553,  5.074, 0.613]
}

def main():
    print(row_border("="))
    print(row_title("UNIFIED PIANC UNDER KEEL CLEARANCE (UKC) CALCULATOR"))
    print(row_border("="))
    
    # -------------------------------------------------------------------------
    # 1. INPUT PHASE
    # Collect principal dimensions, loading condition, channel geometry,
    # manoeuvring inputs, wind data, and wave data used by the calculation chain.
    # -------------------------------------------------------------------------
    v_type = get_input("Vessel type (bulk/container/tanker/gas/cargo)", "bulk").lower()
    L_pp = float(get_input("Length between perpendiculars, L_pp (m)", 213.0))
    B = float(get_input("Maximum beam, B (m)", 32.3))
    T = float(get_input("Static draft, T (m)", 12.8))
    tonnage = float(get_input("Tonnage (DWT or GT)", 60000))
    load_pct = float(get_input("Vessel Loading Condition (%)", 100.0))
    V_s = float(get_input("Transit speed, V_s (m/s)", 3.86)) # ~7.5 knots
    V_k = V_s * 1.94384 
    
    h = float(get_input("Nominal Water Depth, h (m)", 14.0))
    channel_type = get_input("Channel type (unrestricted/restricted/canal)", "restricted").lower()
    W = float(get_input("Bottom width of channel, W (m)", f"{3.5*B:.2f}"))
    h_T = float(get_input("Trench height (0 if canal/unrestricted), h_T (m)", 0.0))
    n_bank = float(get_input("Inverse bank slope (run/rise), n", 7.0))
    
    V_WR = float(get_input("Relative wind speed, V_WR (m/s)", 10.0))
    theta_WR = float(get_input("Relative wind angle, theta_WR (degrees)", 90.0))
    K_R_turn = float(get_input("Non-dimensional index of turning, K_R", 0.5))
    delta_R = float(get_input("Rudder angle, delta_R (degrees)", 15.0))
    C_phi = float(get_input("Transient heel coefficient, C_phi", 1.5))
    
    H_s = float(get_input("Significant wave height, H_s (m)", 0.5))
    T_wave = float(get_input("Wave period, T_w (s)", 3.0))
    wave_angle = float(get_input("Wave incidence angle, psi (degrees)", 90.0))
    mu_roll = float(get_input("Roll magnification factor, mu", 3.0))
    gamma_slope = float(get_input("Effective wave slope coefficient, gamma", 0.8))
    N_w = float(get_input("Number of waves encountered, N_w", 500))
    P_m = float(get_input("Probability of exceedance, P_m", 0.00115))

    # -------------------------------------------------------------------------
    # 2. DERIVED HYDROSTATICS & ARCHITECTURE
    # Reconstruct hydrostatic, stability, displacement, and windage quantities
    # from the selected vessel type and the entered principal dimensions.
    # -------------------------------------------------------------------------
    cb_avg, gm_ratio, F_K, a_f, b_f, a_l, b_l = VESSEL_DATA.get(v_type, VESSEL_DATA['bulk'])
    
    C_B = cb_avg                                         # Block Coefficient
    C_WP = 0.18 + 0.86 * C_B                             # Waterplane Coefficient approximation
    GM = gm_ratio * T                                    # Metacentric Height (Transverse stability)
    V_disp = C_B * L_pp * B * T                          # Volume Displacement (Nabla)
    Disp_MT = V_disp * 1.025                             # Mass Displacement in Seawater (Delta)
    KB = T * (0.84 - (0.33 * C_B) / (0.18 + 0.87 * C_B)) # Vertical Center of Buoyancy
    BM = (B ** 2) / (20.4 * C_B * T)                     # Metacentric Radius
    KG = KB - GM + BM                                    # Vertical Center of Gravity
    A_VF = a_f * (tonnage ** b_f)                        # Frontal Windage Area (Regression)
    A_VL = a_l * (tonnage ** b_l)                        # Lateral Windage Area (Regression)
    x_L = L_pp / 2.0                                     # Longitudinal center of wind area
    KG_W = T + (A_VL / (2 * L_pp))                       # Windage Centroid Height

    # -------------------------------------------------------------------------
    # 3. CHANNEL BLOCKAGE & KINEMATICS
    # Build the effective flow section, blockage factor, and Froude numbers used
    # by the squat and wave sub-models.
    # -------------------------------------------------------------------------
    g = 9.80665
    A_S = 0.98 * B * T                                   # Immersed Midship Area
    F_nh = V_s / math.sqrt(g * h)                        # Depth Froude Number
    F_nT = V_s / math.sqrt(g * T)                        # Draft Froude Number
    
    if channel_type == 'unrestricted':
        # In unrestricted water the code still uses a finite effective width
        # of hydrodynamic influence rather than an infinite lateral domain.
        W_eff = (7.04 / (C_B**0.85)) * B
        A_C = W_eff * h  # Bank slope (n) is 0 for unrestricted
        W_Top = W_eff
        S = A_S / A_C if A_C > 0 else 0.0
    else:
        # Restricted and canal sections are represented as trapezoidal flow areas.
        A_C = (W * h) + (n_bank * h**2)                  # Channel Cross-Section Area
        W_Top = W + 2 * n_bank * h                       # Channel Width at Water Surface
        S = A_S / A_C if A_C > 0 else 0.0                # Blockage Factor (S)

    # -------------------------------------------------------------------------
    # 4. HYDRODYNAMIC SQUAT MODEL SET (6 FORMULAS)
    # Evaluate the implemented PIANC-style squat methods and later adopt the
    # average of the three highest values as the operational squat allowance.
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # [1] ICORELS (1980) FORMULA
    # Classic PIANC concept-stage squat expression based on displacement and
    # depth Froude number, with a guarded near-critical denominator.
    # -------------------------------------------------------------------------
    # The coefficient C_S depends on the Block Coefficient (C_B) and vessel type
    # according to Finnish Maritime Administration (FMA) and BAW guidelines.
    if v_type == 'container':
        C_S = 2.0  # BAW recommendation for modern larger containerships
    elif C_B < 0.70:
        C_S = 1.7  # FMA standard for slender vessels
    elif 0.70 <= C_B < 0.80:
        C_S = 2.0  # FMA standard for medium block coefficients
    else:
        C_S = 2.4  # FMA standard for full-form vessels (bulk/tankers)
        
    term1_icorels = V_disp / (L_pp**2)
    term2_icorels = (F_nh**2) / math.sqrt(max(0.001, 1 - F_nh**2))
    squat_icorels = C_S * term1_icorels * term2_icorels

    # -------------------------------------------------------------------------
    # [2] RÖMISCH (1989) FORMULA (PIANC Eq 26.19 - 26.29)
    # Detailed design formula calculating both Bow and Stern squat individually.
    # -------------------------------------------------------------------------
    # Hydraulic celerities and depth scales used by the Römisch method.
    C_U = math.sqrt(g * h)    
    h_m = A_C / W_Top if W_Top > 0 else h
    C_m = math.sqrt(g * h_m)    
    h_mT = h - (h_T / h) * (h - h_m) if h > 0 else h
    C_mT = math.sqrt(g * h_mT)    
    
    # Römisch unrestricted, canal, and restricted correction factors.
    K_U = 0.58 * ((h / T) * (L_pp / B))**0.125 if T > 0 and B > 0 else 1.0
    safe_S = min(max(S, 0.0), 0.999) # Bound S to prevent math domain errors on arcsin
    K_C = (2.0 * math.sin(math.asin(1.0 - safe_S) / 3.0))**1.5
    K_R_romisch = K_U * (1.0 - (h_T / h)) + K_C * (h_T / h)
    
    # Critical speed selection depends on the channel representation.
    if channel_type == 'unrestricted':
        V_cr = C_U * K_U
    elif channel_type == 'canal':
        V_cr = C_m * K_C
    else: # restricted
        V_cr = C_mT * K_R_romisch
    speed_ratio = min(0.99, V_s / V_cr) if V_cr > 0 else 0.99    
    
    # Speed, hull-form, and depth multipliers used in bow/stern squat.
    C_V = 8 * (speed_ratio**2) * ((speed_ratio - 0.5)**4 + 0.0625)
    C_F = (10 * C_B / (L_pp / B))**2
    K_delT = 0.155 * math.sqrt(h / T)    
    
    # Evaluate bow and stern squat separately and keep the governing value.
    squat_romisch_bow = C_V * C_F * K_delT * T
    squat_romisch_stern = C_V * K_delT * T
    squat_romisch = max(squat_romisch_bow, squat_romisch_stern)

    # -------------------------------------------------------------------------
    # [3] Millward (1992)
    # Forward and aft sinkage expressions with the larger value adopted as the
    # Millward result.
    # -------------------------------------------------------------------------
    safe_Fnh2 = min(F_nh**2, 1.05) 
    term1_fp = (15 * C_B * (T / L_pp)) - 0.55
    term2_fp = safe_Fnh2 / (1 - 0.9 * safe_Fnh2)
    S_FP = term1_fp * term2_fp * (L_pp / 100.0)
    
    term1_ap = (61.7 * C_B * (T / L_pp)) - 0.6
    term2_ap = safe_Fnh2 / math.sqrt(max(0.001, 1 - safe_Fnh2))
    S_AP = term1_ap * (L_pp / 100.0) * term2_ap
    squat_millward = max(S_FP, S_AP)

    # -------------------------------------------------------------------------
    # [4] ERYUZLU (1994) FORMULA (PIANC Eq 26.14, 26.15)
    # Draft-Froude-based squat relation including a width correction factor.
    # -------------------------------------------------------------------------
    W_B_ratio = W / B if B > 0 else float('inf')    
    # Correction factor for channel width (K_b)
    # Forced to 1.0 for unrestricted channels regardless of W/B ratio
    if channel_type == 'unrestricted' or W_B_ratio >= 9.61:
        K_b = 1.0
    else:
        K_b = 3.1 / math.sqrt(W_B_ratio)      
    # Squat calculation utilizing the Ship Draft Froude number (F_nT)
    term1_eryuzlu = 0.298 * (h**2 / T)
    term2_eryuzlu = F_nT ** 2.289
    term3_eryuzlu = (h / T) ** -2.972    
    squat_eryuzlu = term1_eryuzlu * term2_eryuzlu * term3_eryuzlu * K_b

    # -------------------------------------------------------------------------
    # [5] BARRASS B3 (2004) FORMULA (PIANC Eq 26.9, 26.10, 26.11)
    # Compact concept-stage squat expression using blockage and speed in knots.
    # -------------------------------------------------------------------------
    # Determine the blockage multiplier (K). 
    # If unrestricted or S <= 0.10, K = 1.0. 
    # For restricted channels, K modulates the denominator between 100 and 50,
    # meaning K must be strictly bounded between 1.0 and 2.0.
    if channel_type == 'unrestricted' or S <= 0.10:
        K_barrass = 1.0
    else:
        K_barrass = max(1.0, min(5.74 * (S ** 0.76), 2.0))        
    # S_Max calculates the maximum squat (governs UKC limits)
    squat_barrass = (K_barrass * C_B * (V_k ** 2)) / 100.0    
    # Calculate squat at the "other end" (S_oe) based on Eq 26.11
    K_oe = 1.0 - 40.0 * (0.7 - C_B)**2
    squat_barrass_oe = K_oe * squat_barrass
        
    # -------------------------------------------------------------------------
    # [6] Ankudinov MARSIM (2009)
    # Composite model separating midship sinkage and dynamic trim.
    # -------------------------------------------------------------------------
    # Base hull and Froude parameters
    P_Hu = 1.7 * C_B * ((B * T) / (L_pp**2)) + 0.004 * (C_B**2)
    P_hT = 1.0 + 0.35 / ((h / T)**2)
    P_Fnh = F_nh**(1.8 + 0.4 * F_nh)
    
    # Channel boundary interaction modifiers
    h_T_ratio_ank = h_T / h if channel_type != 'unrestricted' else 0.0
    S_h = C_B * (S / (h / T)) * h_T_ratio_ank if (h/T) > 0 else 0.0
    P_Ch1 = 1.0 + 10 * S_h - 1.5 * (1.0 + S_h) * math.sqrt(S_h)
    P_Ch2 = max(0.0, 1.0 - 5 * S_h)
    
    # Midship sinkage evaluation
    K_p_S = 0.15 
    S_m = (1 + K_p_S) * P_Hu * P_Fnh * P_hT * P_Ch1
    
    # Dynamic Trim Evaluation
    n_Tr = 2.0 + 0.8 * (P_Ch1 / C_B)
    K_p_T, K_b_T, K_v_T, K_T1_T = 0.15, 0.10, 0.04, 0.0 
    K_Tr = (C_B**n_Tr) - (0.15 * K_p_S + K_p_T) - (K_b_T + K_v_T + K_T1_T)
    
    h_T_ratio_calc = h / T
    if h_T_ratio_calc >= 1.0:
        P_hT_trim_modifier = 1.0 - math.exp(-1.2 * (h_T_ratio_calc - 1.0))
    else:
        P_hT_trim_modifier = 0.0
        
    Tr = -1.7 * P_Hu * P_Fnh * P_hT_trim_modifier * K_Tr * P_Ch2
    
    # Total Squat = Midship Sinkage + Half the absolute trim
    squat_ankudinov = L_pp * (S_m + 0.5 * abs(Tr))

    # -------------------------------------------------------------------------
    # Rank the six squat predictions and adopt the arithmetic mean of the
    # three highest values as the conservative operational squat allowance.
    # -------------------------------------------------------------------------
    squat_results = [
        ("ICORELS (1980)", squat_icorels),
        ("Römisch (1989)", squat_romisch),
        ("Millward (1992)", squat_millward),
        ("Eryuzlu (1994)", squat_eryuzlu),
        ("Barrass (2004)", squat_barrass),
        ("Ankudinov (2009)", squat_ankudinov)
    ]
    squat_results.sort(key=lambda x: x[1], reverse=True)
    top_3_squats = squat_results[:3]
    adopted_squat = sum(val for name, val in top_3_squats) / 3.0

    # -------------------------------------------------------------------------
    # 5. DENSITY AND HEEL PENALTIES
    # -------------------------------------------------------------------------
    # [A] Density-related draft increment based on the adopted operational
    # freshwater-type allowance.
    density_penalty = T * 0.025 * (C_B / C_WP)
    
    # [B] Wind Heeling Angle (phi_W)
    rho_a = 1.25     # Air density (kg/m³)
    gamma_w = 10052  # Specific weight of seawater (N/m³)
    
    # Convert displacement mass back to displaced seawater volume for the
    # hydrostatic restoring-moment scale used in wind heel.
    V_disp_vol = (Disp_MT * 1000) / 1025.0
    restoring_moment_factor = gamma_w * V_disp_vol * GM

    theta_WR_rad = math.radians(theta_WR)

    # Yamano & Saito (1997) Regression Coefficients: [C_Yn0, C_Yn1, C_Yn2, C_Yn3, C_Yn4]
    # Empirically calculates the dimensionless aerodynamic lateral force.
    C_Y_coeffs = {
        1: [ 0.509,  4.904,  0.0,    0.0,    0.022],
        2: [ 0.0208, 0.230, -0.075,  0.0,    0.0  ],
        3: [-0.357,  0.943,  0.0,    0.0381, 0.0  ]
    }

    # Assemble the lateral aerodynamic coefficient from the harmonic terms.
    C_Wy = 0.0
    for n in range(1, 4):
        c0, c1, c2, c3, c4 = C_Y_coeffs[n]
        C_Yn = c0 + c1*(A_VL / L_pp**2) + c2*(x_L / L_pp) + c3*(L_pp / B) + c4*(A_VL / A_VF)
        C_Wy += C_Yn * math.sin(n * theta_WR_rad)

    # Lateral wind force and associated heeling moment.
    F_Wy = 0.5 * rho_a * C_Wy * A_VL * (V_WR**2) 
    l_W = KG_W - (T / 2.0)
    M_W = l_W * F_Wy 
    wind_heel_deg = math.degrees(M_W / restoring_moment_factor)

    # [C] Turn-induced heel from steady centrifugal action with transient
    # amplification applied through C_phi.
    delta_R_rad = math.radians(delta_R)
    R_C = L_pp / (K_R_turn * delta_R_rad)                           # Turning radius
    phi_C_rad = ((KG - (T / 2)) * (V_s**2)) / (g * R_C * GM)       # Steady Turn Heel
    turn_heel_deg = math.degrees(C_phi * phi_C_rad)                # Transient over-heel
    
    # [D] Total Bilge Sinkage Penalty
    # Projects the combined angular list into a vertical bilge penetration depth.
    total_dynamic_heel = wind_heel_deg + turn_heel_deg
    bilge_penalty = F_K * (B / 2) * math.sin(math.radians(total_dynamic_heel))

    # -------------------------------------------------------------------------
    # 6. WAVE-INDUCED VERTICAL MOTIONS
    # -------------------------------------------------------------------------
    # Obtain the wavelength from the external nonlinear wave routine used by
    # the second wave method and by the report.
    lambda_w = function.L(H_s, T_wave, h, V_s)

    # -------------------------------------------------------------------------
    # [A] METHOD 1: bounded geometric roll-plus-pitch allowance.
    # -------------------------------------------------------------------------
    Z_phi1 = 0.044 * B
    Z_theta1 = 0.0087 * L_pp
    
    # Combine the roll and pitch surrogates and cap them at 2*H_s.
    z_max1_parts = Z_phi1 + Z_theta1
    z_max1_hs = 2.0 * H_s
    
    z_max1 = min(z_max1_parts, z_max1_hs)

    # -------------------------------------------------------------------------
    # [B] METHOD 2: Japanese Z3-type bilge sinkage allowance driven by wave
    # slope forcing and amplified roll response.
    # -------------------------------------------------------------------------
    z2_chart_sinkage = 0.0 
    
    if lambda_w > 0:
        Phi = 360.0 * ((0.35 * H_s) / lambda_w) * math.sin(math.radians(wave_angle))
    else:
        Phi = 0.0

    phi_Max_deg = mu_roll * gamma_slope * Phi 

    # Unbounded Z3 allowance before application of the physical cap.
    Z_3_unbounded = 0.7 * (H_s / 2.0) + (B / 2.0) * math.sin(math.radians(phi_Max_deg))
    
    # Cap the method to prevent unrealistic growth in very short-wave cases.
    Z_3 = min(Z_3_unbounded, 2.0 * H_s)

    z_max2 = max(z2_chart_sinkage, Z_3)

    # -------------------------------------------------------------------------
    # [C] METHOD 3: ROM 3.1 probabilistic allowance using six correction
    # factors for encounter statistics, ship geometry, speed, depth, and angle.
    # -------------------------------------------------------------------------
    C1 = 0.707 * math.sqrt(math.log(N_w / -math.log(1 - P_m)))
    c2_table = [[0.20, 0.08, 0.04], [0.27, 0.12, 0.06], [0.35, 0.17, 0.10], [0.44, 0.23, 0.14]]
    y_interp = [interp_1d(L_pp, [100.0, 200.0, 300.0], row) for row in c2_table]
    C2 = interp_1d(H_s, [1.0, 1.5, 2.5, 4.0], y_interp)
    C3 = interp_1d(load_pct, [50.0, 90.0], [1.20, 1.00])
    C4 = interp_1d(F_nh, [0.05, 0.25], [1.00, 1.35])
    C5 = interp_1d(h / T, [1.05, 1.50], [1.10, 1.00])
    psi_norm = wave_angle % 180
    if psi_norm > 90: psi_norm = 180 - psi_norm
    C6 = interp_1d(psi_norm, [0, 15, 35, 90], [1.0, 1.0, 1.4, 1.7])
    
    z_max3_unbounded = H_s * C1 * C2 * C3 * C4 * C5 * C6
    z_max3 = min(z_max3_unbounded, 2.0 * H_s)
    
    # Adopt the arithmetic mean of the three wave methods.
    adopted_wave = (z_max1 + z_max2 + z_max3) / 3.0

    # -------------------------------------------------------------------------
    # 7. FINAL UKC SUMMATION AND REPORT GENERATION
    # -------------------------------------------------------------------------
    gross_ukc = h - T
    total_dynamic_sinkage = adopted_squat + density_penalty + bilge_penalty + adopted_wave
    net_ukc = gross_ukc - total_dynamic_sinkage
    effective_depth = h - total_dynamic_sinkage
    effective_depth_ratio = effective_depth / T

    report = []
    
    report.append(row_border("+"))
    report.append(row_title("PIANC UNDER KEEL CLEARANCE (UKC) REPORT"))
    report.append(row_border("+"))
    
    report.append(row_title("1. INPUT PARAMETERS"))
    report.append(row_border("-"))
    report.append(row_3col("Parameter", "Symbol", "Value"))
    report.append(row_border("-"))
    
    report.append(row_1col("[A] VESSEL CHARACTERISTICS"))
    report.append(row_3col("Vessel Type", "Type", v_type.upper()))
    report.append(row_3col("Length Between Perpendiculars", "L_pp", f"{L_pp:.2f} m"))
    report.append(row_3col("Maximum Beam", "B", f"{B:.2f} m"))
    report.append(row_3col("Static Draft", "T", f"{T:.2f} m"))
    report.append(row_3col("Reference Tonnage", "DWT/GT", f"{tonnage:,.0f}"))
    report.append(row_3col("Loading Condition", "Load_%", f"{load_pct:.1f} %"))
    report.append(row_border("-"))
    
    report.append(row_1col("[B] CHANNEL & TRANSIT GEOMETRY"))
    report.append(row_3col("Nominal Water Depth", "h", f"{h:.2f} m"))
    report.append(row_3col("Channel Configuration", "Config", channel_type.capitalize()))
    report.append(row_3col("Channel Bottom Width", "W", f"{W:.2f} m"))
    report.append(row_3col("Trench Height", "h_T", f"{h_T:.2f} m"))
    report.append(row_3col("Inverse Bank Slope", "n", f"{n_bank:.2f}"))
    report.append(row_border("-"))
    
    report.append(row_1col("[C] KINEMATICS & METOCEAN CONDITIONS"))
    report.append(row_3col("Relative Wind Speed", "V_WR", f"{V_WR:.2f} m/s"))
    report.append(row_3col("Relative Wind Angle", "theta_WR", f"{theta_WR:.1f} deg"))
    report.append(row_3col("Turning Index", "K_R", f"{K_R_turn:.2f}"))
    report.append(row_3col("Rudder Angle", "delta_R", f"{delta_R:.1f} deg"))
    report.append(row_3col("Transient Heel Coefficient", "C_phi", f"{C_phi:.2f}"))
    report.append(row_3col("Significant Wave Height", "H_s", f"{H_s:.2f} m"))
    report.append(row_3col("Wave Period", "T_w", f"{T_wave:.1f} s"))
    report.append(row_3col("Fenton Wavelength", "lambda_w", f"{lambda_w:.2f} m"))
    report.append(row_3col("Wave Incidence Angle", "psi", f"{wave_angle:.1f} deg"))
    report.append(row_3col("Roll Magnification Factor", "mu", f"{mu_roll:.2f}"))
    report.append(row_3col("Effective Slope Coefficient", "gamma", f"{gamma_slope:.2f}"))
    report.append(row_3col("Number of Waves Encountered", "N_w", f"{N_w:.0f}"))
    report.append(row_3col("Probability of Exceedance", "P_m", f"{P_m:.5f}"))
    report.append(row_border("+"))

    report.append(row_title("2. DERIVED HYDROSTATICS & KINEMATICS"))
    report.append(row_border("-"))
    report.append(row_3col("Parameter", "Symbol", "Value"))
    report.append(row_border("-"))
    
    report.append(row_1col("[A] HYDROSTATIC COEFFICIENTS & DISPLACEMENT"))
    report.append(row_3col("Block Coefficient", "C_B", f"{C_B:.3f}"))
    report.append(row_3col("Waterplane Coefficient", "C_WP", f"{C_WP:.3f}"))
    report.append(row_3col("Volume Displacement", "Nabla", f"{V_disp:,.1f} m3"))
    report.append(row_3col("Mass Displacement", "Delta", f"{Disp_MT:,.0f} MT"))
    report.append(row_border("-"))
    
    report.append(row_1col("[B] STABILITY & AERODYNAMIC PROFILES"))
    report.append(row_3col("Vertical Center of Buoyancy", "KB", f"{KB:.3f} m"))
    report.append(row_3col("Metacentric Radius", "BM", f"{BM:.3f} m"))
    report.append(row_3col("Vertical Center of Gravity", "KG", f"{KG:.3f} m"))
    report.append(row_3col("Transverse Metacentric Height", "GM", f"{GM:.3f} m"))
    report.append(row_3col("Frontal Windage Area", "A_VF", f"{A_VF:,.1f} m2"))
    report.append(row_3col("Lateral Windage Area", "A_VL", f"{A_VL:,.1f} m2"))
    report.append(row_3col("Windage Centroid Height", "KG_W", f"{KG_W:.3f} m"))
    report.append(row_border("-"))
    
    report.append(row_1col("[C] KINEMATICS & CHANNEL BLOCKAGE"))
    report.append(row_3col("Transit Speed", "V_s", f"{V_s:.2f} m/s ({V_k:.1f} kts)"))
    report.append(row_3col("Depth Froude Number", "F_nh", f"{F_nh:.3f}"))
    report.append(row_3col("Draft Froude Number", "F_nT", f"{F_nT:.3f}"))
    report.append(row_3col("Midship Immersed Area", "A_S", f"{A_S:,.1f} m2"))
    # Handle the display of unrestricted effective boundaries gracefully
    if channel_type == 'unrestricted':
        report.append(row_3col("Effective Channel Width", "W_eff", f"{W_Top:,.1f} m"))
        report.append(row_3col("Effective Cross-Section", "A_C", f"{A_C:,.1f} m2"))
    else:
        report.append(row_3col("Channel Cross-Section Area", "A_C", f"{A_C:,.1f} m2"))        
    report.append(row_3col("Blockage Factor", "S", f"{S:.4f}"))
    report.append(row_border("+"))

    report.append(row_title("3. HYDRODYNAMIC SQUAT (PIANC ANALYSIS)"))
    report.append(row_border("-"))
    
    # [A] ICORELS
    report.append(row_1col("[A] ICORELS (1980) FORMULA"))
    report.append(row_1col(f"    Base Volumetric Term: Vol / L_pp^2 = {V_disp / (L_pp**2):.3f}"))
    report.append(row_2col("    Calculated Squat", f"{squat_icorels:.3f} m"))
    report.append(row_border("-"))
    
    # [B] ROMISCH
    report.append(row_1col("[B] RÖMISCH (1989) FORMULA"))
    report.append(row_1col(f"    Critical Speed (V_cr): {V_cr:.2f} m/s | Speed Ratio: {speed_ratio:.3f}"))
    report.append(row_1col(f"   Factors: C_V = {C_V:.3f}, C_F = {C_F:.3f}, K_delT = {K_delT:.3f}"))
    report.append(row_2col("    Calculated Squat", f"{squat_romisch:.3f} m"))
    report.append(row_border("-"))
    
    # [C] MILLWARD
    report.append(row_1col("[C] MILLWARD (1992) FORMULA"))
    report.append(row_1col(f"    Forward Squat (S_FP): {S_FP:.3f} m | Aft Squat (S_AP): {S_AP:.3f} m"))
    report.append(row_2col("    Calculated Squat (Max of FP/AP)", f"{squat_millward:.3f} m"))
    report.append(row_border("-"))
    
    # [D] ERYUZLU
    report.append(row_1col("[D] ERYUZLU 2 (1994) FORMULA"))
    report.append(row_1col(f"    Width Correction (K_b): {K_b:.3f} | Draft Froude (F_nT): {F_nT:.3f}"))
    report.append(row_2col("    Calculated Squat", f"{squat_eryuzlu:.3f} m"))
    report.append(row_border("-"))
    
    # [E] BARRASS
    report.append(row_1col("[E] BARRASS B3 (2004) FORMULA"))
    report.append(row_1col(f"    Blockage (S): {S:.4f} | Speed (V_k): {V_k:.2f} kts"))
    report.append(row_2col("    Calculated Squat", f"{squat_barrass:.3f} m"))
    report.append(row_border("-"))
    
    # [F] ANKUDINOV MARSIM
    report.append(row_1col("[F] ANKUDINOV MARSIM (2009) FORMULA"))
    report.append(row_1col(f"    Midship Sinkage (S_m): {S_m:.4f} | Dynamic Trim (Tr): {Tr:.4f}"))
    report.append(row_2col("    Calculated Squat", f"{squat_ankudinov:.3f} m"))
    report.append(row_border("-"))
    
    # RANKING
    report.append(row_1col("[G] SQUAT TOURNAMENT RANKING (DESCENDING)"))
    report.append(row_2col(f"    1. {squat_results[0][0]}", f"{squat_results[0][1]:.3f} m"))
    report.append(row_2col(f"    2. {squat_results[1][0]}", f"{squat_results[1][1]:.3f} m"))
    report.append(row_2col(f"    3. {squat_results[2][0]}", f"{squat_results[2][1]:.3f} m"))
    report.append(row_2col(f"    4. {squat_results[3][0]}", f"{squat_results[3][1]:.3f} m"))
    report.append(row_2col(f"    5. {squat_results[4][0]}", f"{squat_results[4][1]:.3f} m"))
    report.append(row_2col(f"    6. {squat_results[5][0]}", f"{squat_results[5][1]:.3f} m"))
    report.append(row_border("-"))
    
    # ADOPTED SQUAT
    report.append(row_2col("-> ADOPTED HYDRODYNAMIC SQUAT (TOP 3 AVG)", f"{adopted_squat:.3f} m"))
    report.append(row_border("+"))

    report.append(row_title("4. ENVIRONMENTAL SINKAGES & HEEL PENALTIES"))
    report.append(row_border("-"))
    
    # [A] WATER DENSITY
    report.append(row_1col(" [A] WATER DENSITY LOSS PENALTY (FRESHWATER ALLOWANCE)"))
    report.append(row_1col(f"     S_density = {T:.2f} * 0.025 * ({C_B:.3f} / {C_WP:.3f})"))
    report.append(row_2col("     Calculated Density Penalty", f"{density_penalty:.3f} m"))
    report.append(row_border("-"))
    
    # [B] DYNAMIC HEEL
    report.append(row_1col(" [B] ASYMMETRIC DYNAMIC HEEL (WIND + TURNING)"))
    report.append(row_1col(f"     Wind Force (F_Wy): {F_Wy:,.0f} N | Moment Arm (l_W): {l_W:.2f} m"))
    report.append(row_1col(f"     Wind Heel Moment (M_W): {M_W:,.0f} N-m"))
    report.append(row_2col("     1. Wind-Induced Heel Angle (phi_W)", f"{wind_heel_deg:.3f} deg"))
    
    report.append(row_1col(f"     Turning Radius (R_C): {R_C:.1f} m | Transient Factor (C_phi): {C_phi:.2f}"))
    report.append(row_2col("     2. Turn-Induced Heel Angle (phi_R)", f"{turn_heel_deg:.3f} deg"))
    
    report.append(row_2col("     Total Combined Heel Angle", f"{total_dynamic_heel:.3f} deg"))
    report.append(row_1col(f"     S_heel = {F_K:.3f} * ({B:.2f} / 2) * sin({total_dynamic_heel:.3f} deg)"))
    report.append(row_2col("     Calculated Bilge Sinkage Penalty", f"{bilge_penalty:.3f} m"))
    report.append(row_border("+"))

    report.append(row_title("5. WAVE-INDUCED VERTICAL MOTIONS"))
    report.append(row_border("-"))
    
    # [A] TRIGONOMETRIC

    report.append(row_1col(" [A] METHOD 1: TRIGONOMETRIC BOUNDED WORST-CASE"))
    report.append(row_1col(f"     Roll Bound (Z_phi): {Z_phi1:.3f} m | Pitch Bound (Z_theta): {Z_theta1:.3f} m"))
    report.append(row_1col(f"     Unbounded Sum: {z_max1_parts:.3f} m | Maximum Bound (2*H_s): {z_max1_hs:.3f} m"))    
    if z_max1_parts > z_max1_hs:
        report.append(row_1col("     -> Unbounded sum exceeds physical limit. Applying 2*H_s cap."))        
    report.append(row_2col("     Calculated M1 Allowance", f"{z_max1:.3f} m"))
    report.append(row_border("-"))
    
    # [B] JAPANESE Z3
    report.append(row_1col(" [B] METHOD 2: JAPANESE Z3 BILGE SINKAGE METHOD"))
    report.append(row_1col(f"     Effective Wave Slope (Phi): {Phi:.2f} deg"))
    report.append(row_1col(f"     Resonant Roll (phi_Max) = {mu_roll:.1f} * {gamma_slope:.2f} * {Phi:.2f} = {phi_Max_deg:.2f} deg"))
    if Z_3_unbounded > (2.0 * H_s):
        report.append(row_1col(f"     Unbounded Z3 ({Z_3_unbounded:.3f} m) exceeds physical limit. Capping at 2*H_s."))      
    report.append(row_2col("     Calculated M2 Allowance", f"{z_max2:.3f} m"))
    report.append(row_border("-"))
    
    # [C] ROM 3.1
    report.append(row_1col(" [C] METHOD 3: SPANISH ROM 3.1 SPECTRAL METHOD"))
    report.append(row_1col(f"     C1 (Wave Height) = {C1:.3f} | C2 (Transformation) = {C2:.3f}"))
    report.append(row_1col(f"     C3 (Load Factor) = {C3:.3f} | C4 (Speed Factor) = {C4:.3f}"))
    report.append(row_1col(f"     C5 (Depth Factor) = {C5:.3f}| C6 (Angle Factor) = {C6:.3f}"))    
    if z_max3_unbounded > (2.0 * H_s):
        report.append(row_1col(f"     Unbounded M3 ({z_max3_unbounded:.3f} m) exceeds physical limit. Capping at 2*H_s."))        
    report.append(row_2col("     Calculated M3 Allowance", f"{z_max3:.3f} m"))
    report.append(row_border("-"))
    
    # ADOPTED WAVE
    report.append(row_2col("-> ADOPTED WAVE ALLOWANCE (AVERAGE)", f"{adopted_wave:.3f} m"))
    report.append(row_border("+"))

    report.append(row_title("6. FINAL UNDER KEEL CLEARANCE (UKC) SUMMATION"))
    report.append(row_border("-"))
    report.append(row_2col("Gross UKC (h - T)", f"{gross_ukc:>8.3f} m"))
    report.append(row_border("-"))
    report.append(row_2col("Adopted Hydrodynamic Squat", f"{adopted_squat:>8.3f} m"))
    report.append(row_2col("Adopted Wave Allowance", f"{adopted_wave:>8.3f} m"))
    report.append(row_2col("Water Density Penalty", f"{density_penalty:>8.3f} m"))
    report.append(row_2col("Dynamic Bilge Heel Penalty", f"{bilge_penalty:>8.3f} m"))
    report.append(row_border("-"))
    report.append(row_2col("Total Dynamic Sinkage (Delta T_dyn)", f"{total_dynamic_sinkage:>8.3f} m"))
    report.append(row_2col("FINAL NET UKC (Clearance to Seabed)", f"{net_ukc:>8.3f} m"))
    report.append(row_2col("Effective Water Depth (D_eff)", f"{effective_depth:>8.3f} m"))
    report.append(row_2col("Effective Depth to Draft Ratio (D_eff / T)", f"{effective_depth_ratio:>8.3f}"))
    report.append(row_border("+"))

    # -------------------------------------------------------------------------
    # 8. DIAGNOSTIC AND MINIMUM SAFE NOMINAL DEPTH SEARCH
    #
    # Minimum-depth solver consistent with the same sinkage model used by the
    # main report:
    #   - Squat is recomputed at each depth using all 6 formulas, then the
    #     adopted squat is the average of the top 3 values (tournament method).
    #   - Wave allowance is recomputed at each depth using the same 3 methods,
    #     then adopted as their average.
    #   - Density and heel penalties remain fixed within this depth search.
    #
    # The reported recommended depth is the MINIMUM VERIFIED SAFE depth on the
    # reporting increment grid (step), meaning: (h_req - sinkage(h_req)) / T >= 1.10.
    # -------------------------------------------------------------------------
    diag_status = ""
    diag_rec = ""
    if effective_depth_ratio < 1.10:
        diag_status = "CRITICAL / UNSAFE"

        target_ratio = 1.10
        safety_margin = 0.01         # conservative bracketing target
        target_safe = target_ratio + safety_margin

        step = 0.10                  # reporting / stepping increment [m]
        max_step_ups = 4000          # hard cap to avoid infinite loops
        max_bisect = 120
        tol_h = 0.0005               # bisection tolerance [m]

        def evaluate_at_depth(h_test: float):
            """
            Recompute ALL depth-dependent quantities using the same formulas as the report
            and return (sinkage_total, ratio, adopted_squat_local, adopted_wave_local).
            """
            # --- Channel blockage & kinematics at h_test (same logic as Section 3) ---
            F_nh_test = V_s / math.sqrt(g * h_test)
            F_nT_test = V_s / math.sqrt(g * T)

            if channel_type == "unrestricted":
                W_eff_test = (7.04 / (C_B**0.85)) * B
                A_C_test = W_eff_test * h_test
                W_Top_test = W_eff_test
                S_test = A_S / A_C_test if A_C_test > 0 else 0.0
            else:
                A_C_test = (W * h_test) + (n_bank * h_test**2)
                W_Top_test = W + 2 * n_bank * h_test
                S_test = A_S / A_C_test if A_C_test > 0 else 0.0

            # --- Squat formula set (same formulas/structure as Section 4) ---

            # [1] ICORELS (1980)
            # NOTE: Uses the same previously selected C_S logic and same terms.
            term1_icorels_local = V_disp / (L_pp**2)
            term2_icorels_local = (F_nh_test**2) / math.sqrt(max(0.001, 1 - F_nh_test**2))
            squat_icorels_local = C_S * term1_icorels_local * term2_icorels_local

            # [2] RÖMISCH (1989)
            C_U_local = math.sqrt(g * h_test)
            h_m_local = A_C_test / W_Top_test if W_Top_test > 0 else h_test
            C_m_local = math.sqrt(g * h_m_local)
            h_mT_local = h_test - (h_T / h_test) * (h_test - h_m_local) if h_test > 0 else h_test
            C_mT_local = math.sqrt(g * h_mT_local)

            K_U_local = 0.58 * ((h_test / T) * (L_pp / B)) ** 0.125 if T > 0 and B > 0 else 1.0
            safe_S_local = min(max(S_test, 0.0), 0.999)
            K_C_local = (2.0 * math.sin(math.asin(1.0 - safe_S_local) / 3.0)) ** 1.5
            K_R_romisch_local = K_U_local * (1.0 - (h_T / h_test)) + K_C_local * (h_T / h_test)

            if channel_type == "unrestricted":
                V_cr_local = C_U_local * K_U_local
            elif channel_type == "canal":
                V_cr_local = C_m_local * K_C_local
            else:
                V_cr_local = C_mT_local * K_R_romisch_local

            speed_ratio_local = min(0.99, V_s / V_cr_local) if V_cr_local > 0 else 0.99
            C_V_local = 8 * (speed_ratio_local**2) * ((speed_ratio_local - 0.5) ** 4 + 0.0625)
            C_F_local = (10 * C_B / (L_pp / B)) ** 2
            K_delT_local = 0.155 * math.sqrt(h_test / T)

            squat_romisch_bow_local = C_V_local * C_F_local * K_delT_local * T
            squat_romisch_stern_local = C_V_local * K_delT_local * T
            squat_romisch_local = max(squat_romisch_bow_local, squat_romisch_stern_local)

            # [3] Millward (1992)
            safe_Fnh2_local = min(F_nh_test**2, 1.05)
            term1_fp_local = (15 * C_B * (T / L_pp)) - 0.55
            term2_fp_local = safe_Fnh2_local / (1 - 0.9 * safe_Fnh2_local)
            S_FP_local = term1_fp_local * term2_fp_local * (L_pp / 100.0)

            term1_ap_local = (61.7 * C_B * (T / L_pp)) - 0.6
            term2_ap_local = safe_Fnh2_local / math.sqrt(max(0.001, 1 - safe_Fnh2_local))
            S_AP_local = term1_ap_local * (L_pp / 100.0) * term2_ap_local
            squat_millward_local = max(S_FP_local, S_AP_local)

            # [4] ERYUZLU (1994)
            W_B_ratio_local = W / B if B > 0 else float("inf")
            if channel_type == "unrestricted" or W_B_ratio_local >= 9.61:
                K_b_local = 1.0
            else:
                K_b_local = 3.1 / math.sqrt(W_B_ratio_local)

            term1_ery_local = 0.298 * (h_test**2 / T)
            term2_ery_local = F_nT_test**2.289
            term3_ery_local = (h_test / T) ** -2.972
            squat_eryuzlu_local = term1_ery_local * term2_ery_local * term3_ery_local * K_b_local

            # [5] BARRASS (2004)
            if channel_type == "unrestricted" or S_test <= 0.10:
                K_barrass_local = 1.0
            else:
                K_barrass_local = max(1.0, min(5.74 * (S_test**0.76), 2.0))

            squat_barrass_local = (K_barrass_local * C_B * (V_k**2)) / 100.0

            # [6] Ankudinov MARSIM (2009)
            P_Hu_local = 1.7 * C_B * ((B * T) / (L_pp**2)) + 0.004 * (C_B**2)
            P_hT_local = 1.0 + 0.35 / ((h_test / T) ** 2)
            P_Fnh_local = F_nh_test ** (1.8 + 0.4 * F_nh_test)

            h_T_ratio_ank_local = h_T / h_test if channel_type != "unrestricted" else 0.0
            S_h_local = C_B * (S_test / (h_test / T)) * h_T_ratio_ank_local if (h_test / T) > 0 else 0.0
            P_Ch1_local = 1.0 + 10 * S_h_local - 1.5 * (1.0 + S_h_local) * math.sqrt(S_h_local)
            P_Ch2_local = max(0.0, 1.0 - 5 * S_h_local)

            K_p_S_local = 0.15
            S_m_local = (1 + K_p_S_local) * P_Hu_local * P_Fnh_local * P_hT_local * P_Ch1_local

            n_Tr_local = 2.0 + 0.8 * (P_Ch1_local / C_B)
            K_p_T_local, K_b_T_local, K_v_T_local, K_T1_T_local = 0.15, 0.10, 0.04, 0.0
            K_Tr_local = (C_B**n_Tr_local) - (0.15 * K_p_S_local + K_p_T_local) - (
                K_b_T_local + K_v_T_local + K_T1_T_local
            )

            h_T_ratio_calc_local = h_test / T
            if h_T_ratio_calc_local >= 1.0:
                P_hT_trim_modifier_local = 1.0 - math.exp(-1.2 * (h_T_ratio_calc_local - 1.0))
            else:
                P_hT_trim_modifier_local = 0.0

            Tr_local = -1.7 * P_Hu_local * P_Fnh_local * P_hT_trim_modifier_local * K_Tr_local * P_Ch2_local
            squat_ankudinov_local = L_pp * (S_m_local + 0.5 * abs(Tr_local))

            squat_pack = [
                ("ICORELS (1980)", squat_icorels_local),
                ("Römisch (1989)", squat_romisch_local),
                ("Millward (1992)", squat_millward_local),
                ("Eryuzlu (1994)", squat_eryuzlu_local),
                ("Barrass (2004)", squat_barrass_local),
                ("Ankudinov (2009)", squat_ankudinov_local),
            ]
            squat_pack.sort(key=lambda x: x[1], reverse=True)
            top3_local = squat_pack[:3]
            adopted_squat_local = sum(v for _, v in top3_local) / 3.0

            # --- Wave allowance at h_test (same structure as the main run) ---
            lambda_w_local = function.L(H_s, T_wave, h_test, V_s)
            if isinstance(lambda_w_local, str):
                lambda_w_local = 0.0

            # Method 1: unchanged by depth
            z_max1_local = z_max1

            # Method 2: depends on lambda
            if lambda_w_local > 0:
                Phi_local = 360.0 * ((0.35 * H_s) / lambda_w_local) * math.sin(math.radians(wave_angle))
            else:
                Phi_local = 0.0
            phi_Max_deg_local = mu_roll * gamma_slope * Phi_local
            Z_3_unbounded_local = 0.7 * (H_s / 2.0) + (B / 2.0) * math.sin(math.radians(phi_Max_deg_local))
            Z_3_local = min(Z_3_unbounded_local, 2.0 * H_s)
            z_max2_local = max(0.0, Z_3_local)

            # Method 3: ROM 3.1 depends on F_nh and depth factor C5(h/T)
            C4_local = interp_1d(F_nh_test, [0.05, 0.25], [1.00, 1.35])
            C5_local = interp_1d(h_test / T, [1.05, 1.50], [1.10, 1.00])
            z_max3_unbounded_local = H_s * C1 * C2 * C3 * C4_local * C5_local * C6
            z_max3_local = min(z_max3_unbounded_local, 2.0 * H_s)

            adopted_wave_local = (z_max1_local + z_max2_local + z_max3_local) / 3.0

            # Total sinkage & ratio
            sinkage_total = adopted_squat_local + density_penalty + bilge_penalty + adopted_wave_local
            ratio = (h_test - sinkage_total) / T

            return sinkage_total, ratio, adopted_squat_local, adopted_wave_local

        # --- Bracket: step UP until VERIFIED safe under the SAME model ---
        h_unsafe = max(0.01, h)
        _, r_unsafe, _, _ = evaluate_at_depth(h_unsafe)

        if r_unsafe >= target_safe:
            h_safe = h_unsafe
            _, r_safe, _, _ = evaluate_at_depth(h_safe)
        else:
            h_safe = max(h_unsafe, target_safe * T)
            h_safe = math.ceil(h_safe / step) * step
            _, r_safe, _, _ = evaluate_at_depth(h_safe)

            step_ups = 0
            while r_safe < target_safe and step_ups < max_step_ups:
                h_unsafe = h_safe
                r_unsafe = r_safe
                h_safe += step
                _, r_safe, _, _ = evaluate_at_depth(h_safe)
                step_ups += 1

            if r_safe < target_safe:
                diag_rec = (
                    "Ratio < 1.10. Solver could not find a verified safe depth within limits. "
                    f"Last tested h={h_safe:.2f} m gave ratio={r_safe:.3f}."
                )

        # --- Refine: bisection between last unsafe and first safe ---
        # IMPORTANT: diag_rec is always defined (initialized above). Use a value check.
        if not diag_rec:
            lo = h_unsafe
            hi = h_safe

            for _ in range(max_bisect):
                mid = 0.5 * (lo + hi)
                _, r_mid, _, _ = evaluate_at_depth(mid)
                if (hi - lo) <= tol_h:
                    break
                if r_mid >= target_safe:
                    hi = mid
                else:
                    lo = mid

            # Round UP and VERIFY strictly on the step grid against target_ratio (1.10)
            eps = 1e-12
            h_req = math.ceil((hi + eps) / step) * step

            sink_req, ratio_req, squat_req, wave_req = evaluate_at_depth(h_req)
            guard = 0
            while ratio_req < target_ratio and guard < max_step_ups:
                h_req += step
                sink_req, ratio_req, squat_req, wave_req = evaluate_at_depth(h_req)
                guard += 1

            # -----------------------------------------------------------------
            # STRUCTURAL SAFETY GUARD
            # Recommended depth must NEVER be less than current nominal depth.
            # If convergence yields a lower value (numerical artefact), force
            # h_req >= h (rounded up to step) and re-evaluate.
            # -----------------------------------------------------------------
            if h_req < h:
                h_req = math.ceil(h / step) * step
                sink_req, ratio_req, squat_req, wave_req = evaluate_at_depth(h_req)

            # Force internal consistency of summary variables at the recommended depth
            total_dynamic_sinkage = sink_req
            effective_depth = h_req - sink_req
            effective_depth_ratio = ratio_req

            # -------------------------------------------------------------
            # Final Recommendation Logic (Ratio < 1.10 branch)
            # -------------------------------------------------------------
            diag_status = "CRITICAL / UNSAFE"

            if ratio_req < target_ratio:
                diag_rec = (
                    "Ratio < 1.10. Unable to verify a compliant minimum depth "
                    f"within solver limits. Last evaluated depth h = {h_req:.2f} m "
                    f"produced D_eff/T = {ratio_req:.3f}. Further depth increase required."
                )
            else:
                diag_rec = (
                    f"Ratio < 1.10. Required minimum nominal depth (h) >= {h_req:.2f} m."
                )

    elif 1.10 <= effective_depth_ratio < 1.25:
        diag_status = "MARGINAL (CALM WATERS ONLY)"
        diag_rec = (
            "No additional depth required. Effective depth ratio satisfies minimum PIANC threshold (D_eff/T ≥ 1.10). Operational caution advised."
        )

    elif 1.25 <= effective_depth_ratio < 1.50:
        diag_status = "STANDARD / SAFE"
        diag_rec = (
            "No additional depth required. Adequate navigational clearance for normal operational conditions."
        )

    else:
        diag_status = "HIGH CLEARANCE"
        diag_rec = (
            "No additional depth required. Clearance exceeds conservative "
            "PIANC design thresholds."
        )
		
    # -----------------------------------------------------------------
    # DIAGNOSIS & RECOMMENDATION (wrapped to avoid truncation)
    # -----------------------------------------------------------------
    if not diag_rec or str(diag_rec).strip() == "":
        diag_rec = "No additional depth required."
    report.extend(row_1col_wrap(f" DIAGNOSIS      : {diag_status}"))
    report.extend(row_1col_wrap(f" RECOMMENDATION : {diag_rec}", indent=16))
    report.append(row_border("+"))


    final_report = "\n".join(report)
    
    # Output to console
    print(final_report)
    
    # Export securely to file
    with open("output.txt", "w", encoding="utf-8") as file:
        file.write(final_report)
    print(f"\n[SUCCESS] Report saved to {os.path.abspath('output.txt')}")
# =============================================================================
# TKINTER GUI WRAPPER
# =============================================================================
# The GUI is a presentation layer around the same calculation path used by the
# CLI. It only gathers inputs, calls main(), and displays the generated report.
#
# Usage:
#   - GUI: python navigation-calculator.py --gui
#   - CLI: python navigation-calculator.py --cli
#   - Auto: if stdin is not a TTY (e.g., double-click on Windows), GUI launches.
# =============================================================================

def _run_main_with_input_sequence(responses):
    """
    Execute main() while feeding a predefined sequence of responses to input().

    This preserves the original input prompts and calculation path exactly.
    The patched input() deliberately does NOT print prompts, so the captured
    output contains only the program's explicit prints (banner + report).
    """
    import builtins
    import contextlib
    import io
    import traceback

    it = iter(responses)

    def _fake_input(_prompt=""):
        try:
            return next(it)
        except StopIteration:
            # Force get_input() to fall back to defaults if we run out.
            raise EOFError

    buf = io.StringIO()
    old_input = builtins.input
    builtins.input = _fake_input
    try:
        with contextlib.redirect_stdout(buf):
            main()
    except Exception:
        # Re-raise with captured stdout for easier diagnostics.
        captured = buf.getvalue()
        tb = traceback.format_exc()
        raise RuntimeError(
            "Calculation failed.\n\n--- Captured stdout ---\n"
            + captured
            + "\n\n--- Traceback ---\n"
            + tb
        ) from None
    finally:
        builtins.input = old_input

    return buf.getvalue()


def _extract_report_text(captured_stdout):
    """
    Extract the framed PIANC report from captured stdout.

    The CLI prints a small banner first and then prints the full report, followed
    by a [SUCCESS] line. This function returns only the report.
    """
    lines = str(captured_stdout).splitlines()

    start = 0
    for i, line in enumerate(lines):
        if "PIANC UNDER KEEL CLEARANCE (UKC) REPORT" in line:
            start = max(0, i - 1)
            break

    end = len(lines)
    for i, line in enumerate(lines):
        if line.strip().startswith("[SUCCESS]"):
            end = i
            break

    report_text = "\n".join(lines[start:end]).strip("\n")
    return report_text


def run_gui():
    """Launch the Tkinter GUI front-end."""
    try:
        import tkinter as tk
        import tkinter.font as tkfont
        from tkinter import ttk, filedialog, messagebox
        from tkinter.scrolledtext import ScrolledText
    except Exception as exc:
        raise RuntimeError(
            "Tkinter is not available in this Python environment. "
            "Install/enable Tkinter or run the CLI mode with --cli."
        ) from exc

    # ---- Defaults (match CLI defaults exactly where applicable) ----
    DEFAULTS = {
        "v_type": "bulk",
        "L_pp": "213.0",
        "B": "32.3",
        "T": "12.8",
        "tonnage": "60000",
        "load_pct": "100.0",
        "V_s": "3.86",
        "h": "14.0",
        "channel_type": "restricted",
        "W": f"{3.5 * 32.3:.2f}",
        "h_T": "0.0",
        "n_bank": "7.0",
        "V_WR": "10.0",
        "theta_WR": "90.0",
        "K_R": "0.5",
        "delta_R": "15.0",
        "C_phi": "1.5",
        "H_s": "0.5",
        "T_wave": "3.0",
        "wave_angle": "90.0",
        "mu_roll": "3.0",
        "gamma_slope": "0.8",
        "N_w": "500",
        "P_m": "0.00115",
    }
    DEFAULT_W_B_RATIO = "3.5"

    INPUT_ORDER = [
        ("v_type", "Vessel type (bulk/container/tanker/gas/cargo)"),
        ("L_pp", "Length between perpendiculars, L_pp (m)"),
        ("B", "Maximum beam, B (m)"),
        ("T", "Static draft, T (m)"),
        ("tonnage", "Tonnage (DWT or GT)"),
        ("load_pct", "Vessel Loading Condition (%)"),
        ("V_s", "Transit speed, V_s (m/s)"),
        ("h", "Nominal Water Depth, h (m)"),
        ("channel_type", "Channel type (unrestricted/restricted/canal)"),
        ("W", "Bottom width of channel, W (m)"),
        ("h_T", "Trench height (0 if canal/unrestricted), h_T (m)"),
        ("n_bank", "Inverse bank slope (run/rise), n"),
        ("V_WR", "Relative wind speed, V_WR (m/s)"),
        ("theta_WR", "Relative wind angle, theta_WR (degrees)"),
        ("K_R", "Non-dimensional index of turning, K_R"),
        ("delta_R", "Rudder angle, delta_R (degrees)"),
        ("C_phi", "Transient heel coefficient, C_phi"),
        ("H_s", "Significant wave height, H_s (m)"),
        ("T_wave", "Wave period, T_w (s)"),
        ("wave_angle", "Wave incidence angle, psi (degrees)"),
        ("mu_roll", "Roll magnification factor, mu"),
        ("gamma_slope", "Effective wave slope coefficient, gamma"),
        ("N_w", "Number of waves encountered, N_w"),
        ("P_m", "Probability of exceedance, P_m"),
    ]

    def _safe_str(var):
        s = var.get() if var is not None else ""
        return str(s).strip()

    def _sync_channel_width(*_):
        """Update the derived channel bottom width from beam and W/B ratio."""
        try:
            beam = float(_safe_str(vars_map["B"]))
            ratio = float(_safe_str(w_b_ratio_var))
            width = beam * ratio
        except Exception:
            computed_w_var.set("Enter numeric B and W/B ratio")
            vars_map["W"].set("")
            return

        vars_map["W"].set(f"{width:.2f}")
        computed_w_var.set(f"{width:.2f} m")

    def _set_defaults():
        for key, value in DEFAULTS.items():
            vars_map[key].set(value)
        w_b_ratio_var.set(DEFAULT_W_B_RATIO)
        computed_w_var.set("")
        out_path_var.set("")
        _sync_channel_width()
        nb.select(tab_vessel)

    def _choose_mono_family():
        families = set(tkfont.families())
        for candidate in ("Cascadia Mono", "Consolas", "Lucida Console", "Courier New", "TkFixedFont"):
            if candidate in families:
                return candidate
        return "Courier"

    # ---- Root window and style ----
    root = tk.Tk()
    root.title("PIANC UKC Calculator")
    root.geometry("1220x900")
    root.minsize(1060, 780)

    base_font = ("Segoe UI", 11)
    root.option_add("*Font", base_font)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure("TNotebook", tabposition="n")
    style.configure("TNotebook.Tab", padding=(16, 8), font=("Segoe UI", 11))
    style.configure("TLabel", font=("Segoe UI", 11))
    style.configure("Hint.TLabel", font=("Segoe UI", 10), foreground="#5f6b7a")
    style.configure("Section.TLabel", font=("Segoe UI Semibold", 12))
    style.configure("Title.TLabel", font=("Segoe UI Semibold", 13))
    style.configure("TButton", padding=(10, 6), font=("Segoe UI", 11))
    style.configure("TEntry", padding=4)
    style.configure("TCombobox", padding=4)
    style.configure("TLabelframe", padding=(12, 8))
    style.configure("TLabelframe.Label", font=("Segoe UI Semibold", 11))

    # ---- Variables ----
    vars_map = {key: tk.StringVar(value=value) for key, value in DEFAULTS.items()}
    w_b_ratio_var = tk.StringVar(value=DEFAULT_W_B_RATIO)
    computed_w_var = tk.StringVar(value="")
    out_path_var = tk.StringVar(value="")

    vars_map["B"].trace_add("write", _sync_channel_width)
    w_b_ratio_var.trace_add("write", _sync_channel_width)

    # ---- Main layout ----
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)

    intro = ttk.Frame(root, padding=(14, 12, 14, 6))
    intro.grid(row=0, column=0, sticky="ew")
    intro.columnconfigure(0, weight=1)

    ttk.Label(intro, text="PIANC Under Keel Clearance Calculator", style="Title.TLabel").grid(
        row=0, column=0, sticky="w"
    )
    ttk.Label(
        intro,
        text=(
            "Enter vessel, channel, metocean, and manoeuvre inputs. "
            "The GUI feeds the same calculation engine used by CLI mode."
        ),
        style="Hint.TLabel",
        wraplength=1040,
        justify="left",
    ).grid(row=1, column=0, sticky="w", pady=(4, 0))

    nb = ttk.Notebook(root)
    nb.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

    # ---- Tab containers ----
    tab_vessel = ttk.Frame(nb, padding=14)
    tab_channel = ttk.Frame(nb, padding=14)
    tab_metocean = ttk.Frame(nb, padding=14)
    tab_manoeuvre = ttk.Frame(nb, padding=14)
    tab_run = ttk.Frame(nb, padding=14)

    nb.add(tab_vessel, text="Vessel")
    nb.add(tab_channel, text="Channel")
    nb.add(tab_metocean, text="Metocean")
    nb.add(tab_manoeuvre, text="Manoeuvre")
    nb.add(tab_run, text="Run / Report")

    def _make_form_frame(parent, title, note=None):
        parent.columnconfigure(0, weight=1)
        frame = ttk.LabelFrame(parent, text=title)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)
        if note:
            ttk.Label(frame, text=note, style="Hint.TLabel", wraplength=760, justify="left").grid(
                row=0, column=0, columnspan=3, sticky="w", padx=4, pady=(0, 10)
            )
        return frame, (1 if note else 0)

    def _add_row(parent, row, label, textvariable, hint="", values=None, readonly=False):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(4, 10), pady=6)

        if values is not None:
            widget = ttk.Combobox(parent, textvariable=textvariable, values=values, state="readonly")
        else:
            widget_state = "readonly" if readonly else "normal"
            widget = ttk.Entry(parent, textvariable=textvariable, state=widget_state)
        widget.grid(row=row, column=1, sticky="ew", padx=(0, 10), pady=6)

        if hint:
            ttk.Label(parent, text=hint, style="Hint.TLabel", wraplength=300, justify="left").grid(
                row=row, column=2, sticky="w", pady=6
            )
        return widget

    # ---- Vessel tab ----
    vessel_frame, vessel_row = _make_form_frame(
        tab_vessel,
        "Principal particulars",
        "Define the vessel and transit parameters used throughout the UKC assessment.",
    )
    _add_row(vessel_frame, vessel_row + 0, "Vessel type", vars_map["v_type"], "PIANC vessel family", values=list(VESSEL_DATA.keys()))
    _add_row(vessel_frame, vessel_row + 1, "Length between perpendiculars, L_pp (m)", vars_map["L_pp"])
    _add_row(vessel_frame, vessel_row + 2, "Maximum beam, B (m)", vars_map["B"])
    _add_row(vessel_frame, vessel_row + 3, "Static draft, T (m)", vars_map["T"])
    _add_row(vessel_frame, vessel_row + 4, "Tonnage (DWT or GT)", vars_map["tonnage"])
    _add_row(vessel_frame, vessel_row + 5, "Vessel loading condition (%)", vars_map["load_pct"])
    _add_row(vessel_frame, vessel_row + 6, "Transit speed, V_s (m/s)", vars_map["V_s"], "Approx. 7.5 kn = 3.86 m/s")

    # ---- Channel tab ----
    channel_frame, channel_row = _make_form_frame(
        tab_channel,
        "Channel geometry",
        "The GUI derives the channel bottom width W from beam B and the entered W/B ratio.",
    )
    _add_row(channel_frame, channel_row + 0, "Nominal water depth, h (m)", vars_map["h"])
    _add_row(
        channel_frame,
        channel_row + 1,
        "Channel type",
        vars_map["channel_type"],
        values=["unrestricted", "restricted", "canal"],
    )
    _add_row(channel_frame, channel_row + 2, "Channel width ratio, W/B", w_b_ratio_var, "Default ratio = 3.5")
    _add_row(channel_frame, channel_row + 3, "Computed channel bottom width, W (m)", computed_w_var, "Derived automatically from B × (W/B)", readonly=True)
    _add_row(channel_frame, channel_row + 4, "Trench height, h_T (m)", vars_map["h_T"], "Use 0 for canal or unrestricted")
    _add_row(channel_frame, channel_row + 5, "Inverse bank slope (run/rise), n", vars_map["n_bank"])

    # ---- Metocean tab ----
    metocean_frame, metocean_row = _make_form_frame(
        tab_metocean,
        "Environmental conditions",
        "Wind and wave inputs used by the heel and wave-motion allowances.",
    )
    _add_row(metocean_frame, metocean_row + 0, "Relative wind speed, V_WR (m/s)", vars_map["V_WR"])
    _add_row(metocean_frame, metocean_row + 1, "Relative wind angle, theta_WR (deg)", vars_map["theta_WR"])
    _add_row(metocean_frame, metocean_row + 2, "Significant wave height, H_s (m)", vars_map["H_s"])
    _add_row(metocean_frame, metocean_row + 3, "Wave period, T_w (s)", vars_map["T_wave"])
    _add_row(metocean_frame, metocean_row + 4, "Wave incidence angle, psi (deg)", vars_map["wave_angle"])
    _add_row(metocean_frame, metocean_row + 5, "Roll magnification factor, mu", vars_map["mu_roll"])
    _add_row(metocean_frame, metocean_row + 6, "Effective wave slope coefficient, gamma", vars_map["gamma_slope"])
    _add_row(metocean_frame, metocean_row + 7, "Number of waves encountered, N_w", vars_map["N_w"])
    _add_row(metocean_frame, metocean_row + 8, "Probability of exceedance, P_m", vars_map["P_m"])

    # ---- Manoeuvre tab ----
    manoeuvre_frame, manoeuvre_row = _make_form_frame(
        tab_manoeuvre,
        "Manoeuvre inputs",
        "Turning inputs used by the dynamic heel and bilge penetration assessment.",
    )
    _add_row(manoeuvre_frame, manoeuvre_row + 0, "Non-dimensional index of turning, K_R", vars_map["K_R"])
    _add_row(manoeuvre_frame, manoeuvre_row + 1, "Rudder angle, delta_R (deg)", vars_map["delta_R"])
    _add_row(manoeuvre_frame, manoeuvre_row + 2, "Transient heel coefficient, C_phi", vars_map["C_phi"])

    # ---- Run / report tab ----
    tab_run.columnconfigure(0, weight=1)
    tab_run.rowconfigure(3, weight=1)

    run_intro = ttk.Label(
        tab_run,
        text=(
            "Run the UKC calculation and review the fixed-width engineering report below. "
            "The output font is auto-fitted so the 100-character report width stays aligned."
        ),
        style="Hint.TLabel",
        wraplength=980,
        justify="left",
    )
    run_intro.grid(row=0, column=0, sticky="w", pady=(0, 8))

    out_frame = ttk.Frame(tab_run)
    out_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
    out_frame.columnconfigure(1, weight=1)

    ttk.Label(out_frame, text="Save report to (optional):").grid(row=0, column=0, sticky="w", padx=(0, 8))
    ttk.Entry(out_frame, textvariable=out_path_var).grid(row=0, column=1, sticky="ew")

    def _browse_out():
        path = filedialog.asksaveasfilename(
            title="Save report as",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            out_path_var.set(path)

    ttk.Button(out_frame, text="Browse…", command=_browse_out).grid(row=0, column=2, padx=(8, 0))

    btn_frame = ttk.Frame(tab_run)
    btn_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
    btn_frame.columnconfigure(10, weight=1)

    last_report = {"text": ""}

    mono_font = tkfont.Font(root=root, family=_choose_mono_family(), size=11)
    report_box = ScrolledText(tab_run, wrap="none", font=mono_font, undo=False)
    report_box.grid(row=3, column=0, sticky="nsew")
    report_box.configure(state="disabled")

    def _fit_report_font(_event=None):
        try:
            available_px = max(420, report_box.winfo_width() - 32)
        except Exception:
            return

        best_size = 8
        for size in range(18, 7, -1):
            mono_font.configure(size=size)
            if mono_font.measure("0" * REPORT_WIDTH) <= available_px:
                best_size = size
                break
        mono_font.configure(size=best_size)

    report_box.bind("<Configure>", _fit_report_font)

    def _run():
        _sync_channel_width()
        if not _safe_str(vars_map["W"]):
            messagebox.showerror(
                "Input error",
                "Beam B and W/B ratio must be numeric so the channel width W can be derived.",
            )
            nb.select(tab_channel)
            return

        responses = []
        for key, _prompt in INPUT_ORDER:
            value = _safe_str(vars_map[key])
            responses.append(value if value != "" else "")

        try:
            captured = _run_main_with_input_sequence(responses)
            report_text = _extract_report_text(captured)
            if not report_text:
                report_text = captured.strip("\n")

            last_report["text"] = report_text
            report_box.configure(state="normal")
            report_box.delete("1.0", "end")
            report_box.insert("1.0", report_text)
            report_box.configure(state="disabled")
            report_box.see("1.0")
            _fit_report_font()

            output_path = _safe_str(out_path_var)
            if output_path:
                with open(output_path, "w", encoding="utf-8") as handle:
                    handle.write(report_text)
        except Exception as exc:
            messagebox.showerror("Calculation error", str(exc))
            return

        nb.select(tab_run)
        if _safe_str(out_path_var):
            messagebox.showinfo("Done", "Calculation completed and report saved.")
        else:
            messagebox.showinfo(
                "Done",
                "Calculation completed. The CLI engine also writes output.txt in the working folder.",
            )

    def _copy():
        text = last_report.get("text", "")
        if not text:
            messagebox.showwarning("Nothing to copy", "Run the calculation first.")
            return
        root.clipboard_clear()
        root.clipboard_append(text)
        messagebox.showinfo("Copied", "Report copied to clipboard.")

    def _save_as():
        text = last_report.get("text", "")
        if not text:
            messagebox.showwarning("Nothing to save", "Run the calculation first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save report as",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        out_path_var.set(path)
        messagebox.showinfo("Saved", "Report saved.")

    ttk.Button(btn_frame, text="Run calculation", command=_run).grid(row=0, column=0, padx=(0, 8))
    ttk.Button(btn_frame, text="Copy to clipboard", command=_copy).grid(row=0, column=1, padx=(0, 8))
    ttk.Button(btn_frame, text="Save As…", command=_save_as).grid(row=0, column=2, padx=(0, 8))
    ttk.Button(btn_frame, text="Reset defaults", command=_set_defaults).grid(row=0, column=3)

    root.bind("<F5>", lambda _event: _run())
    root.bind("<Control-Return>", lambda _event: _run())

    _set_defaults()
    root.after(150, _fit_report_font)
    root.mainloop()

def _should_launch_gui(argv):
    """Decide whether to launch GUI or CLI."""
    a = set(argv or [])
    if "--cli" in a:
        return False
    return True


if __name__ == "__main__":
    import sys
    if _should_launch_gui(sys.argv[1:]):
        run_gui()
    else:
        main()
