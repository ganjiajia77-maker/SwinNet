"""Disable only the upstream TOPO SVG side effects; matching is unchanged."""


def RenderRegion(*args, **kwargs):
    return None


def RenderRegion2(*args, **kwargs):
    return None


def RenderSVG(*args, **kwargs):
    return None
