"""Categorical colours for sample groups.

Up to five groups take the Okabe-Ito palette, as Wong recommends for colour-blind
readers (B. Wong, "Points of view: Color blindness", Nature Methods 8, 441, 2011),
without four of its colours, each of which already says something else:
its orange is the gold of a high level (ΔE 2.9), its grey the "not significant",
"average" and "no data" greys, its black fades to that grey when other groups
are highlighted, and its yellow vanishes on the white plot background.
The first two are its blue and vermillion, a cool against a warm, as two groups
are told apart in scanpy and Tableau (blue and orange) and ggplot2 (salmon and
teal): ΔE 31 apart, and at least 22 under every colour-vision deficiency.
Vermillion is near the comparison's red for deuteranopes; the two meet only
where the group is named beside its colour.
More groups take scanpy's defaults, which leave grey out as well, in scanpy's
own order of use: its 20-colour palette (Vega's category20 adjusted for colour
blindness), Zeileis et al.'s 28 colours, then the 102 colours of "godsnot"
(scanpy.plotting.palettes, BSD-3). Beyond 102 groups the colours repeat.
"""

from __future__ import annotations

OKABE_ITO = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
)

SCANPY_20 = (
    "#1f77b4", "#ff7f0e", "#279e68", "#d62728", "#aa40fc", "#8c564b", "#e377c2",
    "#b5bd61", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5", "#ad494a", "#8c6d31",
)

SCANPY_28 = (
    "#023fa5", "#7d87b9", "#bec1d4", "#d6bcc0", "#bb7784", "#8e063b", "#4a6fe3",
    "#8595e1", "#b5bbe3", "#e6afb9", "#e07b91", "#d33f6a", "#11c638", "#8dd593",
    "#c6dec7", "#ead3c6", "#f0b98d", "#ef9708", "#0fcfc0", "#9cded6", "#d5eae7",
    "#f3e1eb", "#f6c4e1", "#f79cd4", "#7f7f7f", "#c7c7c7", "#1CE6FF", "#336600",
)

SCANPY_102 = (
    "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
    "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF",
    "#997D87", "#5A0007", "#809693", "#6A3A4C", "#1B4400", "#4FC601", "#3B5DFF",
    "#4A3B53", "#FF2F80", "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92",
    "#FF90C9", "#B903AA", "#D16100", "#DDEFFF", "#000035", "#7B4F4B", "#A1C299",
    "#300018", "#0AA6D8", "#013349", "#00846F", "#372101", "#FFB500", "#C2FFED",
    "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09", "#00489C", "#6F0062",
    "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66", "#885578",
    "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
    "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F",
    "#938A81", "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757",
    "#C8A1A1", "#1E6E00", "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C",
    "#772600", "#D790FF", "#9B9700", "#549E79", "#FFF69F", "#201625", "#72418F",
    "#BC23FF", "#99ADC0", "#3A2465", "#922329", "#5B4534", "#FDE8DC", "#404E55",
    "#0089A3", "#CB7E98", "#A4E804", "#324E72",
)


def group_palette(n: int) -> list[str]:
    """``n`` group colours: Okabe-Ito up to 5, then the smallest scanpy palette that fits."""
    for palette in (OKABE_ITO, SCANPY_20, SCANPY_28, SCANPY_102):
        if n <= len(palette):
            return list(palette[:n])
    return [SCANPY_102[index % len(SCANPY_102)] for index in range(n)]
