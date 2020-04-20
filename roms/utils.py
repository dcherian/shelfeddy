from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import octant

xr.set_options(display_style="text")


Vgrid = namedtuple(
    "Vgrid", ["Vtransform", "Vstretching", "theta_s", "theta_b", "hc", "nz"]
)


def calc_depths(
    ini, grid, vgrid,
):
    C = octant.depths.get_Vstretching(vgrid.Vstretching, vgrid.theta_s, vgrid.theta_b)
    s_rho = octant.depths.get_srho(vgrid.nz)
    s_w = octant.depths.get_sw(vgrid.nz + 1)
    ini = ini.copy().assign_coords(
        s_rho=s_rho,
        s_w=s_w,
        Cs_r=C(s_rho),
        Cs_w=C(s_w),
        theta_s=vgrid.theta_s,
        theta_b=vgrid.theta_b,
        Tcline=vgrid.hc,
        hc=vgrid.hc,
        Vtransform=vgrid.Vtransform,
        Vstretching=vgrid.Vstretching,
    )

    depths = octant.depths.get_depths(vgrid.Vtransform, C, grid.h.values, vgrid.hc)
    ini["z_rho"] = (
        ["ocean_time", "s_rho", "eta_rho", "xi_rho"],
        depths(ini.s_rho.values, ini.zeta.values[np.newaxis, :]),
    )
    ini["z_w"] = (
        ["ocean_time", "s_w", "eta_rho", "xi_rho"],
        depths(ini.s_w.values, ini.zeta.values[np.newaxis, :]),
    )
    ini = ini.set_coords(["z_rho", "z_w"])

    plot_s_surfaces(ini, ini.sizes["eta_rho"] // 2)

    return ini


def resize_dataset(ds, nx, ny, nz):
    ds = ds.copy()
    sizes = dict(
        xi_psi=nx - 1,
        xi_rho=nx,
        xi_u=nx - 1,
        xi_v=nx,
        eta_psi=ny - 1,
        eta_rho=ny,
        eta_u=ny,
        eta_v=ny - 1,
        s_rho=nz,
        s_w=nz + 1,
    )

    for dim in ds.dims:
        if dim in sizes:
            ds[dim] = np.arange(ds.sizes[dim])
            ds = ds.reindex({dim: np.arange(sizes[dim])})

    return ds


# remove lon or x
def trim_geo_or_cartesian(ds, spherical):

    ds = ds.copy()
    varnames = set(ds.variables)

    if spherical == 0:
        for varname in varnames:
            if "lat" in varname or "lon" in varname:
                del ds[varname]

    elif spherical == 1:
        for varname in varnames:
            if "x" in varname or "y" in varname:
                del ds[varname]

    return ds


def process_cdl_output(name, nx, ny, nz, spherical, **kwargs):

    ds = (
        xr.open_dataset(name, decode_times=False, **kwargs)
        .pipe(resize_dataset, nx, ny, nz)
        .pipe(trim_geo_or_cartesian, spherical)
        .assign_coords(spherical=spherical)
    )

    for var in ds.variables:
        ds[var].encoding = {}

    return ds


def plot_s_surfaces(ini, iy, ax=None):
    if ax is None:
        ax = plt.gca()
    x, z, s = xr.broadcast(
        ini.x_rho.isel(eta_rho=iy), ini.z_w.isel(eta_rho=iy).squeeze(), ini.s_w
    )
    ax.contour(x, z, s, levels=ini.s_w.values)
    title = ""
    for var in ["Vtransform", "Vstretching", "theta_s", "theta_b", "hc"]:
        if "theta" in var:
            title += f"$\{var}$={ini[var].values} "
        else:
            title += f"{var}={ini[var].values} "
    ax.set_title(title)


def make_grid(grid, x, y):
    ogrid = octant.grid.CGrid(*np.meshgrid(x, y))
    # assign values from octant
    grid = grid.isel(eta_rho=slice(len(y)), xi_rho=slice(len(x)))
    for varname in grid:
        attr = getattr(ogrid, varname, None)
        if attr is not None:
            grid[varname] = grid[varname].copy(data=attr)

    grid["xl"] = x.max()
    grid["el"] = y.max()

    # masks
    for varname in grid:
        if "mask" in varname:
            grid[varname] = xr.ones_like(grid[varname])

    return grid
