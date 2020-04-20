import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xfilter
import octant

import utils

xr.set_options(display_style="text")

outdir = "./"

print(f"Writing to {outdir}\n")

dx = 100
dy = 100

nx = 790 + 2
ny = 400 + 2
nz = 30

spherical = 0

# hours since ... doesn't seem to work.
time_enc = {
    "units": "days since 2001-01-01 00:00:00",
    "calendar": "proleptic_gregorian",
}

print(f"Grid: {nx-2} x {ny-2} x {nz} \n")

dxvec = np.ones((nx,)) * dx
dyvec = np.ones((ny,)) * dy

stretchfactorx = 2.5
stretchfactory = 2.5
Lstretchx = 7e3
Lstretchy = 0
stretch = {
    "north": False,
    "south": False,
    "east": False,
    "west": True,
}
if stretch["west"]:
    maxdx = stretchfactorx * dx
    nstrx = int(Lstretchx / dx)
    dxvec[:nstrx] = maxdx - (maxdx - dx) / nstrx * np.arange(nstrx)
    assert dxvec[0] == maxdx
    assert dxvec[nstrx] == dx

x = np.insert(np.cumsum(dxvec), 0, 0)
y = np.insert(np.cumsum(dyvec), 0, 0)


print("Making grid...")
grid = utils.process_cdl_output("./grd_spherical.nc", nx, ny, nz, spherical)
grid = utils.make_grid(grid, x, y)

# sponges
Lsponge = 7e3
sponges = {
    "east": False,
    "west": True,
    "north": False,
    "south": False,
}

nspx = int(Lsponge / dx)
nspy = int(Lsponge / dy)
visc_factor = xr.zeros_like(grid.visc_factor)

if sponges["west"]:
    visc_factor = visc_factor.where(
        grid.x_rho > Lsponge, np.abs(Lsponge - grid.x_rho) / Lsponge
    )
visc_factor = visc_factor.clip(max=1) * 1
grid["visc_factor2"] = visc_factor
grid["visc_factor4"] = xr.ones_like(visc_factor)
del grid["visc_factor"]
grid["diff_factor"][:] = 1

# f
grid["f"][:] = 1e-4

# h
Lshelf = 40e3
maxh = 300
shelfmask = grid.x_rho > (grid.xl - Lshelf)
shelf = 10 + (grid.x_rho - grid.xl) * -2e-3
slope = 0 + (grid.x_rho - grid.xl) * -8e-3
shelf = shelf.where(shelfmask)
slope = slope.where(~shelfmask)

dh = slope.min() - shelf.max()
slope -= dh

h = shelf.fillna(0) + slope.fillna(0)

grid["h"] = h.where(h < maxh).fillna(maxh)
smooth = grid.h.copy(deep=True)
# only smooth deep ocean end
for ii in range(30):
    smooth = smooth.rolling(xi_rho=11, center=True, min_periods=1).mean()
smooth[:, -40:] = grid.h[:, -40:]
grid["h"] = smooth

shelf[1, :].plot()
slope[1, :].plot()
grid.h[1, :].plot()

del grid["ZoBot"]
del grid["rdrag"]
del grid["rdrag2"]
del grid["wtype_grid"]

print("Writing grid")
grid.to_netcdf(f"{outdir}/ocean_grd.nc")

# ---------------------------------- forcing
print("Making forcing...")

time = pd.date_range("2009-01-01", "2009-12-31 23:59:00", freq="3H")
frc = utils.process_cdl_output("./frc_uvstress.nc", nx, ny, nz, spherical)

# remove xi, eta dimensions. This should tell roms to apply wind stress uniformly
for name in frc.dims:
    if "xi" in name or "eta" in name:
        frc = frc.isel({name: 0}, drop=True)

frc = frc.squeeze().drop("sms_time").expand_dims(sms_time=time)
dt = frc.sms_time.copy(
    data=(frc.sms_time - frc.sms_time[0])
    .values.astype("timedelta64[h]")
    .astype(np.float32)
)

buoy_stress = True
if buoy_stress:
    buoy = xr.open_dataset("../data/buoy_46050.nc").sel(time="2009")

    kwargs = dict(coord="time", freq=1 / 3, cycles_per="D", debug=False, num_discard=0)
    filtered = xr.Dataset()
    filtered["taux"] = xfilter.lowpass(buoy.taux, **kwargs,)[::3]
    filtered["tauy"] = xfilter.lowpass(buoy.tauy, **kwargs,)[::3]
    filtered.taux[:16] = np.nan
    filtered.tauy[:16] = np.nan
    filtered.taux[0] = 0
    filtered.tauy[0] = 0
    filtered = filtered.interpolate_na("time", method="linear")
    filtered.taux.sel(time="2009").plot()
    filtered.tauy.sel(time="2009").plot()

    filtered = filtered.rename({"time": "sms_time"}).reset_coords(drop=True)

    frc["sustr"] = filtered.taux
    frc["svstr"] = filtered.tauy

    frc.sms_time.attrs["cycle_length"] = 365.0

    frc.attrs = buoy.attrs
    frc.attrs["xfilter"] = str(kwargs)

else:
    period = 7  # days
    strmax = -0.05  # N/m²
    frc["sustr"] = xr.zeros_like(frc.sustr)
    frc["svstr"] = frc.svstr.copy(
        data=(strmax * np.sin(2 * np.pi * dt / 24 / period)).broadcast_like(frc.svstr)
    )
    envelope = np.tanh((dt) / 24 / period)
    envelope -= envelope[0]
    assert (envelope >= 0).all()

    # (envelope * frc.svstr)[:,1,1].sel(sms_time=slice('2001-01-14')).plot()
    # (envelope * strmax).sel(sms_time=slice('2001-01-14')).plot()
    # frc["svstr"] = xr.full_like(frc.svstr, -0.1/1025)

    frc = frc * envelope
    frc["envelope"] = envelope
    frc["period"] = period
    frc["period"].attrs = {"long_name": "Wind stress forcing period", "units": "days"}
    frc["strmax"] = strmax
    frc["strmax"].attrs = {
        "long_name": "Amplitude of sinusoidal wind stress forcing",
        "units": "N/m²",
    }

frc.sms_time.encoding = time_enc
print("Writing forcing")
frc.reset_coords(drop=True).to_netcdf(f"{outdir}/ocean_frc.nc")

# ------------------------------------ initial conditions
print("Making initial")
ini = utils.process_cdl_output("./ini_hydro.nc", nx, ny, nz, spherical)

ini = ini.assign_coords(
    {k: grid.variables[k] for k in grid.variables if "x_" in k or "y_" in k}
)
ini["zeta"][:] = 0

vgrid = utils.Vgrid(Vtransform=2, Vstretching=4, theta_s=0.5, theta_b=2, hc=150, nz=nz)
ini = utils.calc_depths(ini, grid, vgrid)

ini["u"][:] = 0
ini["v"][:] = 0  # -0.01
ini["ubar"][:] = 0
ini["vbar"][:] = 0  # -0.01
# del ini["salt"]

ini["temp"] = ini.temp.copy(data=5 * (1 - np.tanh(-(ini.z_rho + 30) / 35))) + 10
N2 = 9.81 * 1.7e-4 * ini.temp.diff("s_rho") / ini.z_rho.diff("s_rho")
print(f"{N2.min().item():.2e} ≤ N² ≤ {N2.max().item():.2e}")

perturb = ini.temp.copy(data=np.random.randn(*ini.temp.shape))
perturb /= perturb.max()
ini["temp"] += perturb * 1e-3
ini["zeta"] += perturb.isel(s_rho=-1, drop=True) * 1e-4

ini.temp.squeeze().sel(eta_rho=100).plot.contourf(y="z_rho", robust=True)
plt.figure()
ini.temp[:, :, 10, 10].plot()


if (N2 < 0).any().item():
    raise ValueError("N² < 0!")

if (N2 > 1e-2).any().item():
    raise ValueError("N² > 1e-1!")

ini["N2"] = N2
ini.N2.attrs = {"long_name": "N²", "units": "s^{-2}"}

ini["ocean_time"] = ini.ocean_time.copy(data=frc.sms_time[[0]].values)
ini.ocean_time.attrs = {}
ini.ocean_time.encoding = time_enc

print("Writing initial")
ini.reset_coords().to_netcdf(f"{outdir}/ocean_ini.nc")


# ------------------------------ boundary
print("Making boundary")
bry = utils.process_cdl_output("./bry_unlimit.nc", nx, ny, nz, spherical).rename(
    {"bry_time": "ocean_time"}
)
bry = bry.assign_coords(
    {k: ini[k] for k in ini.coords if "x_" not in k and "y_" not in k and "z_" not in k}
)
bry = bry.set_coords("h")

obc = {
    "north": False,
    "south": False,
    "east": False,
    "west": True,
}

eta = ["eta_psi", "eta_rho", "eta_u", "eta_v"]
xi = ["xi_psi", "xi_rho", "xi_u", "xi_v"]
region = {
    "north": dict(zip(eta, [-1,] * 4)),
    "south": dict(zip(eta, [0,] * 4)),
    "east": dict(zip(xi, [-1,] * 4)),
    "west": dict(zip(xi, [0,] * 4)),
}
brynames = set(bry.data_vars)
for varname in brynames:
    field, boundary = varname.split("_")
    if field not in ini or not obc[boundary]:
        print(f"deleting {varname}")
        del bry[varname]
        continue

    iloc = region[boundary]
    possible_dims = set(ini[field].dims) - set(bry[varname].dims)
    dim = [dim for dim in possible_dims if "time" not in dim and "s_" not in dim]
    assert len(dim) == 1
    dim = dim[0]
    bry[varname] = bry[varname].copy(data=ini[field].isel({dim: iloc[dim]}))
    bry[varname].attrs["time"] = "ocean_time"


print("Writing boundary")
# boundary file needs to be as long as forcing?
bry = (
    bry.squeeze()
    .drop("ocean_time")
    .expand_dims(ocean_time=frc.sms_time.values)
    .reset_coords()
)
bry.ocean_time.encoding = time_enc
bry.to_netcdf(f"{outdir}/ocean_bry.nc", unlimited_dims="ocean_time")
