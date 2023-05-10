import numpy as np
import importlib.util

from modules import shared


def get_opt(key, default=False):
    try:
        return shared.opts.__getattr__(key) or default
    except Exception:
        return default

def expand_polygon(points, distance):
    center = np.mean(points, axis=0)
    new_points = []
    for point in points:
        vec = point - center
        vec = vec / np.linalg.norm(vec)
        new_points.append(point + vec * distance)
    return new_points

def calculate_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def index(list, element):
    if element not in list:
        return -1
    return list.index(element)

def removeEmptyStringValues(dct):
    return { k: v for k, v in dct.items() if v }
