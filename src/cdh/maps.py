"""Choropleth maps of exposure and outcome.

Boundaries come from geoBoundaries (gbOpen, CC BY 4.0), which publishes openly
licensed admin boundaries for every country. They are used for DISPLAY only. The
analysis itself is still keyed to whatever geometry the outcome is measured on,
so a name mismatch here degrades the map, never the estimates.

Cartographic choices, made once:

* Choropleths encode magnitude, so every one uses a single-hue sequential ramp,
  light to dark. No rainbow, no diverging scale on a one-directional quantity.
* Districts carry the data; province outlines sit on top in a thin neutral line
  purely for orientation, which is why they are a stroke and not a second fill.
* Projected to an Albers equal-area conic centred on Zimbabwe. Plotting a
  choropleth in unprojected degrees distorts the very areas the reader is
  comparing by eye.
* The colour bar is labelled in real units and its ends are the data's real
  range, so no value in the map falls outside the legend.
"""
from __future__ import annotations

import math
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from . import config as C

GEOBOUNDARIES = ("https://media.githubusercontent.com/media/wmgeolab/geoBoundaries"
                 "/main/releaseData/gbOpen/{iso3}/ADM{lvl}"
                 "/geoBoundaries-{iso3}-ADM{lvl}_simplified.geojson")


def country_crs(gdf) -> str:
    """Albers equal-area conic fitted to the country's own extent.

    Derived rather than hardcoded so the same code maps any country. Standard
    parallels sit at one sixth and five sixths of the latitude span, which is the
    conventional choice for minimising areal distortion across a conic.
    """
    x0, y0, x1, y1 = gdf.to_crs(4326).total_bounds
    lat1 = y0 + (y1 - y0) / 6.0
    lat2 = y0 + 5.0 * (y1 - y0) / 6.0
    return (f"+proj=aea +lat_1={lat1:.4f} +lat_2={lat2:.4f} "
            f"+lat_0={(y0 + y1) / 2:.4f} +lon_0={(x0 + x1) / 2:.4f} "
            f"+datum=WGS84 +units=m +no_defs")


def ensure_boundaries(iso3=None, levels=(1, 2), verbose=True):
    """Download the geoBoundaries layers for a country if they are not present.

    geoBoundaries serves these through GitHub's LFS media endpoint, which is why
    the URL is media.githubusercontent.com and not raw.githubusercontent.com: the
    raw host returns a 132-byte LFS pointer, not the data.
    """
    import urllib.request

    iso3 = (iso3 or C.COUNTRY_ISO3).upper()
    out = {}
    for lvl in levels:
        dst = C.BOUNDARY_DIR / f"{iso3.lower()}_adm{lvl}.geojson"
        out[lvl] = dst
        if dst.exists() and dst.stat().st_size > 10_000:
            continue
        url = GEOBOUNDARIES.format(iso3=iso3, lvl=lvl)
        if verbose:
            print(f"   downloading {iso3} ADM{lvl} boundaries ...")
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            if data[:8] == b"version ":
                raise RuntimeError("got an LFS pointer, not the data")
            dst.write_bytes(data)
        except Exception as e:
            if verbose:
                print(f"   ! could not fetch {iso3} ADM{lvl}: {e}")
    return out

def _note(fig):
    """Synthetic-data watermark, top-right. Not top-centre (collides with a
    left-aligned panel-b title on a 2-panel figure) and not bottom-right
    (collides with the colour-bar label there)."""
    fig.text(0.99, 0.985, "SYNTHETIC DATA — not real results", ha="right",
             va="top", fontsize=11, color="#c8622a", style="italic")


INK = "#1c2321"
SURFACE = "#fcfcfb"

# Single-hue sequential ramps. Light end is a fill, not a thin mark, so the pale
# steps are legible here in a way they would not be on a line.
RAMP_BLUE = LinearSegmentedColormap.from_list(
    "cdh_blue", ["#eaf2f7", "#bcd8e7", "#7fb3d1", "#4288b3", "#1f6f9c", "#0c4463"])
RAMP_RUST = LinearSegmentedColormap.from_list(
    "cdh_rust", ["#fbf0e8", "#f2d3ba", "#e3ab82", "#d1854f", "#b96a2f", "#8a4a1c"])

# Diverging ramps for trend maps, where unlike the quantities above the sign is
# informative: a district can be wetting or cooling, not only drying or warming.
# Colour semantics stay consistent with the rest of the paper either way: rust is
# always the "drier / hotter" direction, blue always "wetter / cooler".
RAMP_DIVERGE_RAIN = LinearSegmentedColormap.from_list(
    "cdh_diverge_rain",
    ["#8a4a1c", "#d1854f", "#f2d3ba", "#fcfcfb", "#bcd8e7", "#7fb3d1", "#0c4463"])
RAMP_DIVERGE_TEMP = LinearSegmentedColormap.from_list(
    "cdh_diverge_temp",
    ["#0c4463", "#7fb3d1", "#bcd8e7", "#fcfcfb", "#f2d3ba", "#d1854f", "#8a4a1c"])

# Ordered categorical palette for the agro-ecological stratum map: reuses the
# same two hue families so it reads as part of the same figure set.
CAT_COLORS = ["#0c4463", "#4a8ab0", "#9cc2d8", "#e3ab82", "#8a4a1c"]

# IPC-phase-convention ramp (green -> gold -> orange -> red -> dark maroon),
# for anything encoding raw lean-season IPC phase (not exposure/hazard
# frequency). Deliberately not the plain sequential RAMP_BLUE: the outcome is
# a severity/hazard quantity, and reads instantly to this paper's actual
# audience (FEWS NET analysts, food-security policymakers) in the colour
# scheme they already use, the same way the drought-risk reference figure at
# https://www.mdpi.com/1660-4601/15/10/2306 uses a yellow-to-dark-red hazard
# ramp rather than a neutral one. Anchors are tuned off the published IPC
# 5-phase colours, not a pixel-exact copy.
#
# Chosen for HLS LIGHTNESS strictly decreasing green -> maroon (0.79, 0.80,
# 0.40, 0.32, 0.17 at the anchors, monotonic from phase 2 on -- checked
# programmatically, not eyeballed: an earlier anchor set read as
# "categorical, not sequential" because phase 2/3 were lighter than phase 1,
# so a reader going by darkness alone (print, some CVD, quick skim) could not
# reliably order severity from colour. That mistake would not have been
# caught without explicitly computing lightness per anchor.
#
# Passes CVD separation and the chroma floor; the phase-4/phase-5
# normal-vision floor is 14.9, just under the 15 target, accepted for the
# same reason as before: the colourbar this ramp is always shown with
# carries numeric labels as a secondary encoding (dataviz skill: "legal
# ONLY with secondary encoding: direct labels, gaps, or texture").
RAMP_IPC = LinearSegmentedColormap.from_list(
    "cdh_ipc", ["#69D38C", "#ECB613", "#BD580F", "#8F2514", "#57000E"])


# Suffixes marking an urban council rather than a rural district. These units are
# DROPPED, not normalised away: stripping "Town" from "Shurugwi Town" collides it
# with the rural district "Shurugwi", silently duplicating rows in the join.
URBAN_RE = re.compile(r"\b(?:urban|town|city|municipality)\b", re.I)


def _norm(s: str) -> str:
    """Normalise a district name for matching across sources."""
    s = str(s).lower().strip()
    s = re.sub(r"\b(district|rural district council|rdc)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _norm_base(s: str) -> str:
    """Normalise AND strip the urban-council suffix, so 'Shurugwi Town' and
    'Shurugwi' collapse to one key. Used only to dissolve urban enclaves into the
    district that surrounds them."""
    return _norm(URBAN_RE.sub(" ", str(s)))


def load_boundaries(adm2_path=None, adm1_path=None, urban="dissolve"):
    """Return (districts, provinces) as GeoDataFrames in the project CRS.

    `urban` controls what happens to the urban councils that geoBoundaries splits
    out as separate ADM2 units:
      "dissolve" (default) merges each into the district of the same name. This is
          display-only and is what produces a map without enclave holes; a hole in
          a choropleth reads as missing data, which these are not.
      "drop" removes them, leaving holes.
      "keep" leaves them as separate units.
    """
    import geopandas as gpd

    iso = C.COUNTRY_ISO3.lower()
    if adm2_path is None or adm1_path is None:
        paths = ensure_boundaries()
        adm2_path = adm2_path or paths.get(2, C.BOUNDARY_DIR / f"{iso}_adm2.geojson")
        adm1_path = adm1_path or paths.get(1, C.BOUNDARY_DIR / f"{iso}_adm1.geojson")
    if not adm2_path.exists():
        raise FileNotFoundError(
            f"{adm2_path} not found and could not be downloaded. Run fetch_data.sh "
            f"or place the geoBoundaries ADM2 layer there by hand.")

    d = gpd.read_file(adm2_path).rename(columns={"shapeName": "district"})

    if urban == "drop":
        d = d[~d["district"].str.contains(URBAN_RE, na=False)].copy()
        d["_key"] = d["district"].map(_norm)
    elif urban == "dissolve":
        d["_base"] = d["district"].map(_norm_base)
        # Prefer the rural name as the label for each dissolved group
        rural = (d[~d["district"].str.contains(URBAN_RE, na=False)]
                 .drop_duplicates("_base").set_index("_base")["district"])
        d = d.dissolve(by="_base").reset_index()
        d["district"] = d["_base"].map(rural).fillna(
            d["district"].str.replace(URBAN_RE, "", regex=True).str.strip())
        d["_key"] = d["district"].map(_norm)
    else:
        d["_key"] = d["district"].map(_norm)

    dup = d.loc[d["_key"].duplicated(keep=False), "district"]
    if len(dup):
        raise ValueError(
            f"Boundary names are not unique after normalisation: {sorted(dup)}. "
            f"A duplicated key silently multiplies rows in the join, so this is "
            f"raised rather than warned. Fix _norm or URBAN_RE in maps.py.")
    crs = country_crs(d)
    d = d.to_crs(crs)

    p = None
    if adm1_path.exists():
        p = gpd.read_file(adm1_path).rename(columns={"shapeName": "province"}).to_crs(crs)
    return d, p


def attach(districts, values: pd.DataFrame, name_col="district", verbose=True):
    """Join a district-level value table to the boundary layer by normalised name.

    Reports what failed to match rather than silently dropping it: an unmatched
    district is a hole in the map, and a hole the reader cannot see is worse than
    one they can.
    """
    v = values.copy()
    v["_key"] = v[name_col].map(_norm)
    if v["_key"].duplicated().any():
        clash = sorted(v.loc[v["_key"].duplicated(keep=False), name_col].unique())
        raise ValueError(f"Value table has duplicate district keys: {clash}")

    merged = districts.merge(v.drop(columns=[name_col]), on="_key", how="left")
    if len(merged) != len(districts):
        raise ValueError(f"Join changed row count {len(districts)} -> {len(merged)}; "
                         f"keys are not one-to-one.")

    unmatched_data = sorted(set(v["_key"]) - set(districts["_key"]))
    value_cols = [c for c in v.columns if c not in ("_key", name_col)]
    if verbose:
        matched = int(merged[value_cols].notna().any(axis=1).sum())
        print(f"   Map join: {matched}/{len(districts)} boundary units carry data"
              f"{f'; {len(unmatched_data)} data rows unmatched' if unmatched_data else ''}")
        if unmatched_data:
            back = v.set_index("_key")[name_col].to_dict()
            print(f"     unmatched e.g. {[back[k] for k in unmatched_data[:6]]}")
    return merged


def _place_labels_no_overlap(ax, points, texts, fontsize=12, min_frac=0.05):
    """Draw bbox-style annotation labels at district centroids, then resolve
    any that collide with each other by pushing them directly apart along
    the line between their centres, a few relaxation passes over every pair.

    Needed because annotate_extremes picks districts by VALUE with no regard
    for geography or label width: two extreme districts are sometimes close
    neighbours (Bindura and Uzumba Maramba Pfungwe are the lowest- and
    highest-outcome districts and both sit in the same compact NE cluster of
    small districts), and a long name like "Uzumba Maramba Pfungwe" renders a
    box several times wider than a short one like "Bindura". Each label's
    radius scales with its longest line's character count and is measured
    against the axes' own data extent (ax.get_xlim/ylim), NOT the bounding
    box of just the picked points -- sizing it off the points themselves is
    circular when the colliding points are close together (their own bbox is
    then tiny too, so "9% of that bbox" understates the label's true size on
    the actual map and never detects the overlap it exists to catch). Pairwise
    relaxation (rather than one point reacting to a frozen first point) finds
    the minimal displacement that clears every pair, so a label never travels
    further than it has to. A label that had to move gets a thin leader line
    back to its true centroid; one that didn't move gets none, so the common
    case (well separated districts) looks unchanged.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span = max(x1 - x0, y1 - y0, 1e-6)

    def _radius(text):
        longest = max(len(line) for line in text.split("\n"))
        return span * min_frac * max(1.0, longest / 8.0)

    pos = [[p.x, p.y] for p in points]
    orig = [(p.x, p.y) for p in points]
    radii = [_radius(t) for t in texts]
    n = len(pos)
    for _ in range(60):
        moved_any = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j][0] - pos[i][0]
                dy = pos[j][1] - pos[i][1]
                dist = math.hypot(dx, dy) or 1e-6
                need = radii[i] + radii[j]
                if dist < need:
                    push = (need - dist) / 2
                    ux, uy = dx / dist, dy / dist
                    pos[i][0] -= ux * push
                    pos[i][1] -= uy * push
                    pos[j][0] += ux * push
                    pos[j][1] += uy * push
                    moved_any = True
        if not moved_any:
            break

    for (x, y), text, (ox, oy) in zip(pos, texts, orig):
        moved = abs(x - ox) > 1e-6 or abs(y - oy) > 1e-6
        # Square corners, opaque fill: a printed map label, not a UI tooltip.
        ax.annotate(text, xy=(ox, oy), xytext=(x, y), ha="center", va="center",
                    fontsize=fontsize, color=INK, zorder=5, fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.3", fc=SURFACE, ec="black", lw=0.7),
                    arrowprops=(dict(arrowstyle="-", color=INK, lw=0.8,
                                     shrinkA=0, shrinkB=8) if moved else None))


def choropleth(ax, gdf, column, cmap, label, provinces=None,
               vmin=None, vmax=None, fmt="{:.2f}", title=None,
               annotate_extremes=0, colorbar=True):
    """One choropleth panel. colorbar=False for small multiples that share one
    bar across panels instead of repeating it under each."""
    vals = pd.to_numeric(gdf[column], errors="coerce")
    vmin = float(np.nanmin(vals)) if vmin is None else vmin
    vmax = float(np.nanmax(vals)) if vmax is None else vmax
    fmt = _auto_fmt(vmin, vmax, start=fmt)

    gdf.plot(column=column, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax,
             edgecolor="black", linewidth=0.5,
             missing_kwds={"color": "#e8ebea", "edgecolor": "black",
                           "linewidth": 0.5, "hatch": "///"})
    if provinces is not None:
        provinces.boundary.plot(ax=ax, color="black", linewidth=1.3, zorder=3)

    if annotate_extremes:
        d = gdf.assign(_v=vals).dropna(subset=["_v"])
        pick = pd.concat([d.nlargest(annotate_extremes, "_v"),
                          d.nsmallest(annotate_extremes, "_v")])
        pts = [r.geometry.representative_point() for _, r in pick.iterrows()]
        texts = [f"{r['district']}\n{_fmt_tick(fmt, r['_v'])}" for _, r in pick.iterrows()]
        _place_labels_no_overlap(ax, pts, texts)

    ax.set_axis_off()
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, loc="left", fontsize=17, color=INK, pad=6, fontweight="bold")

    if not colorbar:
        return ax

    sm = mpl.cm.ScalarMappable(cmap=cmap,
                               norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    cb = plt.colorbar(sm, ax=ax, orientation="horizontal",
                      fraction=0.042, pad=0.02, aspect=26)
    cb.set_label(label, fontsize=13.5, color=INK)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=12, colors=INK, length=3)
    cb.set_ticks(np.linspace(vmin, vmax, 5))
    cb.ax.set_xticklabels([_fmt_tick(fmt, t) for t in np.linspace(vmin, vmax, 5)])
    return ax


def choropleth_categorical(ax, gdf, column, categories, colors, legend_title,
                           provinces=None, title=None):
    """One categorical choropleth panel (e.g. agro-ecological stratum)."""
    code = gdf[column].map({c: i for i, c in enumerate(categories)})
    cmap = mpl.colors.ListedColormap(colors[:len(categories)])
    gdf = gdf.assign(_code=code)

    gdf.plot(column="_code", cmap=cmap, vmin=-0.5, vmax=len(categories) - 0.5, ax=ax,
             edgecolor="black", linewidth=0.5,
             missing_kwds={"color": "#e8ebea", "edgecolor": "black",
                           "linewidth": 0.5, "hatch": "///"})
    if provinces is not None:
        provinces.boundary.plot(ax=ax, color="black", linewidth=1.3, zorder=3)

    ax.set_axis_off()
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, loc="left", fontsize=17, color=INK, pad=6, fontweight="bold")

    handles = [mpl.patches.Patch(facecolor=colors[i], edgecolor="black",
                                 label=str(categories[i]))
               for i in range(len(categories))]
    # lower right, not lower left, so this never collides with the scale bar.
    # frameon=True with an opaque face: this legend sits on top of the
    # choropleth itself (not in a margin), and the SE of Zimbabwe is solidly
    # "Driest third" -- dark navy fill under black text with no background
    # was nearly illegible there, and the borderless legend box visually
    # merged into the district boundary lines it overlapped.
    leg = ax.legend(handles=handles, loc="lower right", fontsize=13,
                    frameon=True, title=legend_title, title_fontsize=14)
    leg.get_frame().set_facecolor(SURFACE)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_alpha(0.92)
    return ax


def _auto_fmt(vmin, vmax, n=5, start="{:.0f}"):
    """Pick enough decimal places that the colour-bar ticks are all distinct.

    A bar reading "18, 19, 19, 19, 20" is worse than no bar: it implies the data
    spans one unit when it spans two.
    """
    ticks = np.linspace(vmin, vmax, n)
    for dec in range(0, 4):
        fmt = "{:." + str(dec) + "f}"
        if len({fmt.format(t) for t in ticks}) == n:
            return fmt
    return "{:.3g}"


def _fmt_tick(fmt, x):
    """fmt.format(x), except a value that rounds to zero never prints as
    "-0": a tick a hair below zero (e.g. a district-mean z-score of
    -0.017, an artefact of de-meaning rather than a real negative) rounds
    to -0 under "{:.0f}" and Python's own float formatting keeps that sign,
    which reads as a typo on a submission figure."""
    s = fmt.format(x)
    return s[1:] if s.startswith("-") and float(s) == 0 else s


def _scalebar(ax, gdf, km=100):
    """A scale bar, because a projected map without one invites misreading."""
    km = int(km)
    x0, y0, x1, y1 = gdf.total_bounds
    bx = x0 + (x1 - x0) * 0.06
    by = y0 + (y1 - y0) * 0.06
    ax.plot([bx, bx + km * 1000], [by, by], color=INK, lw=2.2,
            solid_capstyle="butt", zorder=6)
    ax.text(bx + km * 500, by + (y1 - y0) * 0.018, f"{km} km",
            ha="center", va="bottom", fontsize=10.5, color=INK, zorder=6)


def _scale_km(gdf):
    """A round scale-bar length near a fifth of the map width, so the bar suits
    the country rather than assuming one the size of Zimbabwe."""
    x0, _, x1, _ = gdf.total_bounds
    target = (x1 - x0) / 5000.0                      # metres -> km, one fifth
    for k in (10, 25, 50, 100, 200, 250, 500, 1000):
        if k >= target:
            return k
    return 1000


def map_locator(tag="real", synthetic=False):
    """Geographic context only: Africa, neighbours, and administrative units."""
    import geopandas as gpd
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    world_path = C.BOUNDARY_DIR / "ne_110m_admin_0_countries.geojson"
    if not world_path.exists():
        raise FileNotFoundError(f"Locator basemap missing: {world_path}")
    world = gpd.read_file(world_path)
    africa = world[world["CONTINENT"] == "Africa"]
    country = world[world["ADM0_A3"] == C.COUNTRY_ISO3.upper()]
    if country.empty:
        raise ValueError(f"No locator geometry for {C.COUNTRY_ISO3}")
    districts, provinces = load_boundaries(urban="keep")
    fig = plt.figure(figsize=(12.4, 7.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 2.25], hspace=0.18, wspace=0.06)
    ax_africa = fig.add_subplot(grid[0, 0])
    ax_region = fig.add_subplot(grid[1, 0])
    ax_main = fig.add_subplot(grid[:, 1])
    accent = "#b96842"
    africa.plot(ax=ax_africa, color="#e7ebe6", edgecolor="#8d9890", linewidth=0.35)
    country.plot(ax=ax_africa, color=accent, edgecolor=INK, linewidth=0.6)
    x0, y0, x1, y1 = country.total_bounds
    bounds = (x0 - 4, y0 - 4, x1 + 4, y1 + 4)
    ax_africa.add_patch(Rectangle(bounds[:2], bounds[2]-bounds[0], bounds[3]-bounds[1],
                                 fill=False, edgecolor=INK, lw=0.8))
    neighbours = africa.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
    neighbours.plot(ax=ax_region, color="#e7ebe6", edgecolor="#8d9890", linewidth=0.6)
    country.plot(ax=ax_region, color=accent, edgecolor=INK, linewidth=0.8)
    from shapely.geometry import box
    from shapely.ops import unary_union
    adjacent = unary_union(country.geometry).buffer(0.1)
    for _, row in neighbours.iterrows():
        if not row.geometry.intersects(adjacent):
            continue
        visible = row.geometry.intersection(box(*bounds))
        pt = visible.representative_point()
        ax_region.text(pt.x, pt.y, row["NAME"], ha="center", va="center", fontsize=9,
                       color="white" if row["ADM0_A3"] == C.COUNTRY_ISO3.upper() else INK)
    ax_region.set_xlim(bounds[0], bounds[2])
    ax_region.set_ylim(bounds[1], bounds[3])
    districts.plot(ax=ax_main, color="#f1f2eb", edgecolor="#9ba59c", linewidth=0.4)
    if provinces is not None:
        provinces.boundary.plot(ax=ax_main, color=INK, linewidth=1.0)
        for _, row in provinces.iterrows():
            pt = row.geometry.representative_point()
            name = row["province"]
            # City provinces need offset labels and leaders, not overlapping names.
            # Only those two cross other map content, so only they get a plain
            # opaque knockout patch behind the text; the rest are bare labels,
            # not boxed, like every other province name on the map.
            offset = (16, 20) if name == "Harare" else (-38, -22) if name == "Bulawayo" else (0, 0)
            moved = offset != (0, 0)
            ax_main.annotate(name.replace(" ", "\n"), (pt.x, pt.y), xytext=offset,
                             textcoords="offset points", ha="center", va="center", fontsize=10,
                             color=INK,
                             bbox=dict(facecolor="#f1f2eb", edgecolor="none", pad=1) if moved else None,
                             arrowprops=dict(arrowstyle="-", color=INK, lw=0.7) if moved else None)
    _north_arrow(ax_main, x=0.94, y=0.94)
    _scalebar(ax_main, districts, km=100)
    ax_main.legend(handles=[Line2D([0], [0], color=INK, lw=1, label="Province boundary"),
                            Line2D([0], [0], color="#9ba59c", lw=0.6, label="District / council boundary")],
                   loc="lower right", frameon=False, fontsize=9)
    for label, ax in zip("abc", [ax_africa, ax_region, ax_main]):
        ax.set_axis_off()
        ax.text(0, 1, f"{label}.", transform=ax.transAxes, fontsize=14, fontweight="bold")
    if synthetic:
        _note(fig)
    out = C.FIG / f"map0_locator_{tag}.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def map_study_area(panel: pd.DataFrame, tag="real", synthetic=False):
    """Growing-season rainfall climatology: the agro-ecological gradient the whole
    paper rests on, shown once so the reader can hold it while reading the rest."""
    districts, provinces = load_boundaries()
    # NOT mean hot days: the hot-day threshold is each district's OWN 90th
    # percentile, so mean hot days is roughly constant across districts by
    # construction and a map of it shows nothing. Mean growing-season maximum
    # temperature is the quantity with real spatial variation.
    clim = (panel.groupby("district", observed=True)
                 .agg(rain=("season_rain_mm", "mean"),
                      tmax=("tmax_mean", "mean")).reset_index())
    g = attach(districts, clim)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.6))
    choropleth(axes[0], g, "rain", RAMP_BLUE,
               "Mean growing-season rainfall (mm)", provinces,
               fmt="{:.0f}", title="a. Rainfall climatology", annotate_extremes=1)
    choropleth(axes[1], g, "tmax", RAMP_RUST,
               "Mean growing-season daily maximum temperature (\u00b0C)", provinces,
               fmt="{:.1f}", title="b. Growing-season heat", annotate_extremes=1)
    _scalebar(axes[0], g, km=_scale_km(g))
    for ax in axes:
        ax.set_facecolor(SURFACE)
    if synthetic:
        _note(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = C.FIG / f"map1_study_area_{tag}.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def map_exposure_outcome(panel: pd.DataFrame, tag="real", synthetic=False):
    """Where compound seasons happen, and where food insecurity is worst.

    Two panels rather than one bivariate map: a bivariate choropleth is harder to
    read than the comparison it enables, and the reader's actual question is
    whether the two patterns line up.
    """
    districts, provinces = load_boundaries()
    agg = (panel.groupby("district", observed=True)
                .agg(compound_freq=("compound_bin", "mean"),
                     mean_phase=("phase", "mean")).reset_index())
    agg["compound_freq"] *= 100
    g = attach(districts, agg)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.6))
    choropleth(axes[0], g, "compound_freq", RAMP_RUST,
               "Seasons classified compound drought-heat (%)", provinces,
               fmt="{:.0f}", title="a. Compound exposure", annotate_extremes=1)
    choropleth(axes[1], g, "mean_phase", RAMP_IPC,
               "Mean lean-season IPC phase", provinces,
               fmt="{:.2f}", title="b. Food insecurity outcome", annotate_extremes=1)
    _scalebar(axes[0], g, km=_scale_km(g))
    for ax in axes:
        ax.set_facecolor(SURFACE)
    if synthetic:
        _note(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = C.FIG / f"map2_exposure_outcome_{tag}.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def map_hazard_frequency(panel: pd.DataFrame, tag="real", synthetic=False):
    """Drought-only and heat-only frequency, separately.

    Complements map_exposure_outcome's compound-frequency panel: this shows
    where the two component hazards concentrate individually, since a district
    can be drought-prone, heat-prone, both, or neither.
    """
    districts, provinces = load_boundaries()
    agg = (panel.groupby("district", observed=True)
                .agg(drought_freq=("drought_only", "mean"),
                     heat_freq=("heat_only", "mean")).reset_index())
    agg["drought_freq"] *= 100
    agg["heat_freq"] *= 100
    g = attach(districts, agg)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.6))
    choropleth(axes[0], g, "drought_freq", RAMP_BLUE,
               "Seasons in drought alone (%)", provinces,
               fmt="{:.0f}", title="a. Drought-only frequency", annotate_extremes=1)
    choropleth(axes[1], g, "heat_freq", RAMP_RUST,
               "Seasons in heat stress alone (%)", provinces,
               fmt="{:.0f}", title="b. Heat-only frequency", annotate_extremes=1)
    _scalebar(axes[0], g, km=_scale_km(g))
    for ax in axes:
        ax.set_facecolor(SURFACE)
    if synthetic:
        _note(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = C.FIG / f"map3_hazard_frequency_{tag}.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def map_trends(panel: pd.DataFrame, tag="real", synthetic=False):
    """Decadal trend in growing-season rainfall and daily maximum temperature.

    Descriptive context, not the paper's causal claim: the model itself absorbs
    any trend common across districts via the year fixed effect (see CLAUDE.md).
    This just shows the reader whether, and where, the climate record is
    drifting. Diverging ramp because trend sign is informative either way.
    """
    districts, provinces = load_boundaries()

    def _slope_per_decade(g, col):
        x = g["season_year"].to_numpy(dtype=float)
        y = g[col].to_numpy(dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 8:
            return np.nan
        return np.polyfit(x[ok], y[ok], 1)[0] * 10.0

    trend = (panel.groupby("district", observed=True)
                  .apply(lambda g: pd.Series({
                      "rain_trend": _slope_per_decade(g, "season_rain_mm"),
                      "tmax_trend": _slope_per_decade(g, "tmax_mean"),
                  }), include_groups=False)
                  .reset_index())
    g = attach(districts, trend)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.6))
    rv = float(np.nanmax(np.abs(g["rain_trend"]))) or 1.0
    choropleth(axes[0], g, "rain_trend", RAMP_DIVERGE_RAIN,
               "Rainfall trend (mm / decade)", provinces,
               vmin=-rv, vmax=rv, fmt="{:.0f}",
               title="a. Growing-season rainfall trend", annotate_extremes=1)
    tv = float(np.nanmax(np.abs(g["tmax_trend"]))) or 1.0
    choropleth(axes[1], g, "tmax_trend", RAMP_DIVERGE_TEMP,
               "Tmax trend (°C / decade)", provinces,
               vmin=-tv, vmax=tv, fmt="{:.2f}",
               title="b. Growing-season heat trend", annotate_extremes=1)
    _scalebar(axes[0], g, km=_scale_km(g))
    for ax in axes:
        ax.set_facecolor(SURFACE)
    if synthetic:
        _note(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = C.FIG / f"map4_trends_{tag}.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def map_scenario_delta(proj: pd.DataFrame, tag="real", synthetic=False):
    """Projected change in mean lean-season IPC phase by district, one panel
    per CMIP6 scenario, on a shared diverging scale. This IS a district-level
    choropleth deliberately, unlike the raw climate rasters: delta_phase is
    itself a model output evaluated per district (via that district's own
    fitted or stratum coefficients), not a physical field with its own native
    grid, so there is no raster to show instead.
    """
    if proj.empty:
        return None
    districts, provinces = load_boundaries()
    scenarios = sorted(proj["scenario"].unique())

    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.8 * len(scenarios), 6.6),
                             squeeze=False)
    axes = axes[0]
    dv = float(np.nanmax(np.abs(proj["delta_phase"]))) or 1.0
    for ax, scen in zip(axes, scenarios):
        d = proj[proj["scenario"] == scen][["district", "delta_phase"]]
        g = attach(districts, d, verbose=False)
        choropleth(ax, g, "delta_phase", RAMP_DIVERGE_TEMP,
                  "Change in mean lean-season IPC phase", provinces,
                  vmin=-dv, vmax=dv, fmt="{:+.2f}",
                  title=scen.replace("_2050s", " (2050s)").upper(),
                  annotate_extremes=1)
        ax.set_facecolor(SURFACE)
    _scalebar(axes[0], attach(districts, proj[proj["scenario"] == scenarios[0]]
                              [["district", "delta_phase"]], verbose=False),
             km=_scale_km(districts))
    if synthetic:
        _note(fig)
    fig.tight_layout()
    out = C.FIG / f"map_scenario_delta_{tag}.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def map_strata(panel: pd.DataFrame, strat_label: str, tag="real", synthetic=False):
    """Categorical map of the agro-ecological stratification used in the
    heterogeneity analysis (Natural Region if supplied, else aridity tercile).
    """
    if "stratum" not in panel.columns:
        return None
    districts, provinces = load_boundaries()
    s = panel[["district", "stratum"]].dropna().drop_duplicates("district").copy()
    if s.empty:
        return None
    cats = (list(panel["stratum"].cat.categories) if hasattr(panel["stratum"], "cat")
            else sorted(s["stratum"].unique()))
    s["stratum"] = s["stratum"].astype(str)
    cats = [str(c) for c in cats]
    g = attach(districts, s)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    choropleth_categorical(ax, g, "stratum", cats, CAT_COLORS, strat_label, provinces)
    _scalebar(ax, g, km=_scale_km(g))
    ax.set_facecolor(SURFACE)
    if synthetic:
        _note(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = C.FIG / f"map5_strata_{tag}.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def map_period_facets(panel: pd.DataFrame, value_col="phase", period_years=5,
                      agg="mean", cmap=None, label=None, name="period_facets",
                      tag="real", synthetic=False):
    """Small-multiples choropleth: one panel per period, ALL on the same colour
    scale, so the map's geographic pattern can be compared across time directly
    rather than read one map at a time from memory.

    period_years=1 gives one panel per season-year (dense once the record runs
    past about a decade); the default of 5 is more legible for this study's
    ~17-season real record. Either way every panel shares one vmin/vmax and one
    colourbar, which is the point: only the map, not the scale, should change.
    """
    districts, provinces = load_boundaries()
    cmap = cmap or RAMP_BLUE
    label = label or value_col

    d = panel.dropna(subset=[value_col, "season_year"]).copy()
    if d.empty:
        return None
    y0, y1 = int(d["season_year"].min()), int(d["season_year"].max())

    if period_years <= 1:
        d["_period"] = d["season_year"].astype(int)
        period_label = {p: str(p) for p in sorted(d["_period"].unique())}
    else:
        edges = list(range(y0, y1 + 1, period_years))
        edges.append(y1 + 1) if edges[-1] < y1 else edges.__setitem__(-1, y1 + 1)
        codes = pd.cut(d["season_year"], edges, right=False, labels=False)
        d["_period"] = codes
        period_label = {i: (f"{edges[i]}-{edges[i + 1] - 1}" if edges[i + 1] - 1 > edges[i]
                            else str(edges[i]))
                        for i in range(len(edges) - 1)}

    periods = sorted(d["_period"].dropna().unique())
    tables = {}
    for p in periods:
        t = (d[d["_period"] == p].groupby("district", observed=True)[value_col]
               .agg(agg).reset_index().rename(columns={value_col: "_v"}))
        tables[p] = attach(districts, t, verbose=False)
    vals_all = pd.concat([t["_v"] for t in tables.values()])
    if vals_all.notna().sum() == 0:
        return None
    vmin, vmax = float(np.nanmin(vals_all)), float(np.nanmax(vals_all))
    fmt = _auto_fmt(vmin, vmax)

    ncol = min(4, len(periods))
    nrow = int(np.ceil(len(periods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.95 * nrow),
                             squeeze=False)
    axes = axes.flatten()
    for ax, p in zip(axes, periods):
        choropleth(ax, tables[p], "_v", cmap, "", provinces, vmin=vmin, vmax=vmax,
                  title=period_label[int(p)], colorbar=False)
        ax.set_facecolor(SURFACE)
    for ax in axes[len(periods):]:
        ax.axis("off")
    _scalebar(axes[0], tables[periods[0]], km=_scale_km(tables[periods[0]]))

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, ax=list(axes), orientation="horizontal",
                      fraction=0.05, pad=0.015, shrink=0.55, aspect=40)
    cb.set_label(label, fontsize=13, color=INK)
    cb.outline.set_visible(False)
    cb.set_ticks(np.linspace(vmin, vmax, 5))
    cb.ax.set_xticklabels([_fmt_tick(fmt, t) for t in np.linspace(vmin, vmax, 5)])

    if synthetic:
        _note(fig)
    out = C.FIG / f"map_{name}_{tag}.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def _north_arrow(ax, x=0.06, y=0.94, size=0.05):
    """Simple north arrow (triangle + N), axes-fraction coordinates. Plate
    carree at Zimbabwe's small latitude extent distorts "up" from true north
    by only a degree or two, well inside the line width at this scale."""
    ax.annotate("", xy=(x, y), xytext=(x, y - size), xycoords="axes fraction",
               textcoords="axes fraction",
               arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6,
                               shrinkA=0, shrinkB=0))
    ax.text(x, y + 0.012, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=13, fontweight="bold", color=INK)


def map_raster_period_facets(var: str, years, cmap, label: str, name: str,
                             tag="real", synthetic=False):
    """Small-multiples of the actual gridded raster, one native-resolution
    CHIRPS/ERA5-Land pixel grid per growing season -- NOT a district
    choropleth. A choropleth of a continuous field like rainfall or
    temperature implies the value is uniform within a district and jumps
    discontinuously at its border; the real field does neither. District and
    province boundaries are drawn here as black reference lines on top of the
    continuous surface, never as a fill, so they read as administrative
    context rather than as the data itself.

    Requires gee/extract_period_rasters.py to have been run first (writes
    data/processed/rasters/{var}_{year}.tif); years missing a file are
    skipped with a note rather than failing the whole figure.
    """
    import rasterio
    import rasterio.mask
    import geopandas as gpd

    districts, provinces = load_boundaries()
    districts = districts.to_crs(4326)
    provinces = provinces.to_crs(4326) if provinces is not None else None
    country = gpd.GeoDataFrame(geometry=[districts.union_all()], crs=4326)

    rasters = {}
    for y in years:
        f = C.PROC / "rasters" / f"{var}_{y}.tif"
        if not f.exists():
            continue
        with rasterio.open(f) as src:
            arr, transform = rasterio.mask.mask(src, country.geometry, crop=True,
                                                filled=True, nodata=np.nan)
            arr = arr[0]
            left, top = transform * (0, 0)
            right, bottom = transform * (arr.shape[1], arr.shape[0])
            rasters[y] = (arr, (left, right, bottom, top))

    if not rasters:
        print(f"   Raster maps skipped for {var}: no files in "
              f"{C.PROC / 'rasters'} -- run gee/extract_period_rasters.py first.")
        return None

    all_vals = np.concatenate([a[np.isfinite(a)].ravel() for a, _ in rasters.values()])
    # 2nd/98th percentile, not literal min/max: a raster has far more pixels
    # than a district choropleth, so a handful of extreme edge pixels (one
    # wet corner in one season) stretched the whole colour scale pale and
    # made every period look almost the same -- caught by looking at the
    # actual rendered figure, not by any check on the numbers alone.
    vmin, vmax = float(np.nanpercentile(all_vals, 2)), float(np.nanpercentile(all_vals, 98))
    fmt = _auto_fmt(vmin, vmax)

    yrs = [y for y in years if y in rasters]
    ncol = min(4, len(yrs))
    nrow = int(np.ceil(len(yrs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.7 * ncol, 3.95 * nrow),
                             squeeze=False)
    axes = axes.flatten()
    for i, (ax, y) in enumerate(zip(axes, yrs)):
        arr, extent = rasters[y]
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent,
                 origin="upper", interpolation="nearest")
        districts.boundary.plot(ax=ax, color="black", linewidth=0.6, zorder=3)
        if provinces is not None:
            provinces.boundary.plot(ax=ax, color="black", linewidth=1.4, zorder=4)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect(1 / np.cos(np.deg2rad((extent[2] + extent[3]) / 2)))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(str(y), loc="left", fontsize=14, color=INK, pad=6)
        ax.set_facecolor(SURFACE)
        if i == 0:
            _north_arrow(ax)
    for ax in axes[len(yrs):]:
        ax.axis("off")

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(sm, ax=list(axes), orientation="horizontal",
                      fraction=0.05, pad=0.015, shrink=0.55, aspect=40)
    cb.set_label(label, fontsize=13, color=INK)
    cb.outline.set_visible(False)
    cb.set_ticks(np.linspace(vmin, vmax, 5))
    cb.ax.set_xticklabels([_fmt_tick(fmt, t) for t in np.linspace(vmin, vmax, 5)])

    if synthetic:
        _note(fig)
    out = C.FIG / f"map_{name}_{tag}.png"
    fig.savefig(out, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out


def make_all(panel: pd.DataFrame, tag="real", synthetic=False, strat_label=None,
            period_years=5):
    made = []
    try:
        made.append(map_locator(tag, synthetic))
        made.append(map_study_area(panel, tag, synthetic))
        made.append(map_exposure_outcome(panel, tag, synthetic))
        made.append(map_hazard_frequency(panel, tag, synthetic))
        made.append(map_trends(panel, tag, synthetic))
        if strat_label:
            made.append(map_strata(panel, strat_label, tag, synthetic))
        # Continuous exposure variables BEFORE any drought/heat classification
        # is applied to them -- rainfall, temperature, and their standardised
        # (z-score) forms are continuous physical quantities and are mapped as
        # such here. The binary is_drought/is_hot_season/compound_bin maps
        # below are a derived simplification for the M5 categorical model;
        # showing only those would let a reader see the classification
        # without ever seeing the continuous data it was classified from.
        # Rainfall and temperature are raw physical fields with a real native
        # pixel grid available (CHIRPS/ERA5-Land) -- mapped from that raster
        # directly (see map_raster_period_facets), not as a district
        # choropleth, because the field does not actually jump at a district
        # border the way a choropleth implies. Requires
        # gee/extract_period_rasters.py to have been run; skips with a
        # printed note (not an error) if it has not.
        years = sorted(panel["season_year"].unique().tolist())
        made.append(map_raster_period_facets(
            "rain", years, RAMP_BLUE, "Growing-season rainfall (mm)",
            "rainfall_periods", tag, synthetic))
        made.append(map_raster_period_facets(
            "tmax", years, RAMP_RUST,
            "Growing-season mean daily maximum temperature (°C)",
            "temperature_periods", tag, synthetic))
        made.append(map_period_facets(
            panel, value_col="dry_z", period_years=period_years, cmap=RAMP_BLUE,
            label="Moisture deficit (SPEI-6, reversed, z)", name="moisture_deficit_periods",
            tag=tag, synthetic=synthetic))
        made.append(map_period_facets(
            panel, value_col="heat_z", period_years=period_years, cmap=RAMP_RUST,
            label="Growing-season hot days (z)", name="heat_periods",
            tag=tag, synthetic=synthetic))

        # Classified/derived versions, for direct comparison against the
        # continuous maps above.
        made.append(map_period_facets(
            panel, value_col="phase", period_years=period_years, cmap=RAMP_IPC,
            label="Mean lean-season IPC phase", name="phase_periods",
            tag=tag, synthetic=synthetic))
        made.append(map_period_facets(
            panel, value_col="compound_bin", period_years=period_years, cmap=RAMP_RUST,
            label="Share of seasons compound drought-heat", name="compound_periods",
            tag=tag, synthetic=synthetic))
        made = [m for m in made if m is not None]
        print(f"   {len(made)} maps written to {C.FIG}")
    except FileNotFoundError as e:
        print(f"   Maps skipped: {e}")
    except Exception as e:
        print(f"   Maps failed: {type(e).__name__}: {e}")
    return made
