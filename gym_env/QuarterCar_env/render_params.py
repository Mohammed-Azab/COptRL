
# Render defaults 
RENDER_Y_SCALE   = 5      # vertical exaggeration applied to all z-deflections
RENDER_HIST_SECS = 5.0     # seconds of rolling history in the time-series panel

# Set to False to hide the time-series panel and use full width for the schematic.
RENDER_SHOW_TIMESERIES = True
# Number of time-series subplots shown (1-4): z, z_ddot, F_D, speed.
RENDER_N_TIMESERIES    = 4

# Schematic layout - heights in draw-space (metres, before y-scale is applied).
# z_B and z_W are measured from static equilibrium (= 0), so only deflections
# get multiplied by RENDER_Y_SCALE; these nominal offsets stay fixed.
RENDER_Y_W_NOM = 2.0    # wheel-centre draw height at equilibrium
RENDER_Y_B_NOM = 4.0    # body-centre draw height at equilibrium
RENDER_H_MW    = 0.45   # m_W block height
RENDER_W_MW    = 1.75   # m_W block width
RENDER_H_MB    = 0.45   # m_B block height
RENDER_W_MB    = 1.75  # m_B block width  (wider than m_W)
RENDER_XLIM    = ( -3.0,  15.0)   # x-axis: metres relative to car
RENDER_YLIM    = ( -2.0,   8.5)   # y-axis: draw units
RENDER_ROAD_HALF = 15.0   # road sampled ±this distance from car (m)
RENDER_ROAD_N    = 300    # number of road sample points
RENDER_GROUND_Y  = 1.5    # draw-space offset: shifts ground line + both masses up together

#  Render appearance — colours ─
RENDER_C_MB     = '#f5c842'   # sprung mass body (golden yellow)
RENDER_C_MW     = '#4a86c8'   # unsprung mass (steel blue)
RENDER_C_SPRING = '#e05a1c'   # spring coils (orange-red)
RENDER_C_DAMPER = '#4a86c8'   # damper cylinder / piston (steel blue)
RENDER_C_ROAD   = '#aaaaaa'   # road profile line (light gray)
RENDER_C_GROUND = '#222222'   # ground symbol and contact stem (near-black)

#  Render appearance — spring geometry 
RENDER_SP_X = -0.42   # spring centre x in draw-space
RENDER_SP_W =  0.18   # coil half-amplitude (zigzag width)
RENDER_SP_N =  8      # number of zigzag coil pairs

#  Render appearance — damper geometry 
# Piston-cylinder style: open-top cylinder (⊔) linked to lower mass via a short
# rod (RENDER_DA_LOWER_STEM); piston linked to upper mass via upper rod.
RENDER_DA_X          =  0.6   # damper centre x in draw-space
RENDER_DA_W          =  0.5   # cylinder full width (wall-to-wall), draw-space
RENDER_DA_PIST_H     =  0.3   # piston height, draw-space (fixed, not scaled)
RENDER_DA_PIST_FRAC  =  0.48  # piston_top = y_lower + gap * this  (0→bottom, 1→top)
RENDER_DA_LOWER_STEM =  0.20  # rod length from lower mass to cylinder base

# Fixed cylinder heights sized from nominal mass-to-mass gaps:
#   susp nominal gap = Y_B_NOM − H_MB/2 − Y_W_NOM − H_MW/2 = 4.0−0.225−2.0−0.175 = 1.6
#   tire nominal gap = Y_W_NOM − H_MW/2                     = 2.0−0.175            = 1.825
RENDER_DA_CYL_H_SUSP = 0.88   # suspension cylinder height (≈ 1.6 × 0.55)
RENDER_DA_CYL_H_TIRE = 0.8   # tire cylinder height       (≈ 1.825 × 0.55)

#  Render appearance — contact geometry ─
RENDER_CONTACT_STEM = 0   # short stem length above ground line to contact dot
Y_LINE_OFFSET = 0.7   # vertical offset to shift ground line and contact point up together
