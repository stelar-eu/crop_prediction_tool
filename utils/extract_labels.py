import os
import cv2
import numpy as np
import geopandas as gpd
import pyproj
from shapely.ops import transform
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration ---------------------------------------------------------------
# Crop categories, exactly once here.
# -----------------------------------------------------------------------------
CROP_CATEGORIES_ORDERED = [
    ("ALFALFA", [3301090301]),
    ("BEET", [3301050000, 3301290200, 3301290400]),
    ("CLOVER", [3301090303]),
    ("FLAX", [3301060701, 3301060702]),
    ("FLOWERING_LEGUMES", [3301020700]),
    ("FLOWERS", [3301080000]),
    ("FOREST", [
        3306000000, 3306010000, 3306020000, 3306030000, 3306040000,
        3306050000, 3306060000, 3306070000, 3306080000, 3306980000,
        3306990000,
    ]),
    ("GRAIN_MAIZE", [3301010600, 3301010699]),
    ("GRASSLAND", [3302000000]),
    ("HOPS", [3301060200]),
    ("LEGUMES", [
        3301020100, 3301020500, 3301020600, 3301029900,
        3301090300, 3301090302, 3301090304, 3301090305, 3301090398,
    ]),
    ("NA", [
        3000000000, 3300000000, 3301000000, 3301010000, 3301010100,
        3301010200, 3301010300, 3301010400, 3301010500, 3301010800,
        3301010900, 3301011000, 3301020000, 3301060400, 3301060700,
        3301090000,
    ]),
    ("PERMANENT_PLANTATIONS", [3303010000, 3303060000]),
    ("PLASTIC", [3305000000]),
    ("POTATO", [3301030000]),
    ("PUMPKIN", [3301140400]),
    ("RICE", [3301010700, 3301010799]),
    ("SILAGE_MAIZE", [3301090400]),
    ("SOY", [3301160000]),
    ("SPRING_BARLEY", [3301010402]),
    ("SPRING_OAT", [3301010502]),
    ("SPRING_OTHER_CEREALS", [
        3301011102, 3301011202, 3301011302, 3301011502, 3301011503,
    ]),
    ("SPRING_RAPESEED", [3301060402, 3301060403]),
    ("SPRING_RYE", [3301010302]),
    ("SPRING_SORGHUM", [3301010902]),
    ("SPRING_SPELT", [3301011002]),
    ("SPRING_TRITICALE", [3301010802]),
    ("SPRING_WHEAT", [3301010102, 3301010202]),
    ("SUGARBEET", [3301290700]),
    ("SUNFLOWER", [3301060500]),
    ("SWEET_POTATOES", [3301040000]),
    ("TEMPORARY_GRASSLAND", [
        3301090100, 3301090200, 3301090201, 3301090202, 3301090203,
        3301090204, 3301090205, 3301090206, 3301090207, 3301090208,
        3301090209,
    ]),
    ("WINTER_BARLEY", [3301010401]),
    ("WINTER_OAT", [3301010501]),
    ("WINTER_OTHER_CEREALS", [
        3301011101, 3301011201, 3301011301, 3301011501,
    ]),
    ("WINTER_RAPESEED", [3301060401]),
    ("WINTER_RYE", [3301010301]),
    ("WINTER_SORGHUM", [3301010901]),
    ("WINTER_SPELT", [3301011001]),
    ("WINTER_TRITICALE", [3301010801]),
    ("WINTER_WHEAT", [3301010101, 3301010201]),
]

# Flatten the category lists so we can quickly test membership.
ALL_CROPS_FLAT = [item for _, sublist in CROP_CATEGORIES_ORDERED for item in sublist]

# Optional: set this to ensure shapely uses GEOS instead of PyGEOS if needed.
os.environ.setdefault("USE_PYGEOS", "0")

# -----------------------------------------------------------------------------
# Helper functions ------------------------------------------------------------
# -----------------------------------------------------------------------------

def _get_color_map():
    """Return a static 8‑bit color map identical to the original script."""
    return {
        0: [0, 0, 0],
        1: [0, 255, 0],
        2: [0, 0, 255],
        3: [255, 255, 0],
        4: [255, 165, 0],
        5: [255, 0, 255],
        6: [0, 255, 255],
        7: [128, 0, 128],
        8: [128, 128, 0],
        9: [0, 128, 0],
        10: [128, 0, 0],
        11: [0, 0, 128],
        12: [128, 128, 128],
        13: [0, 128, 128],
        14: [255, 0, 0],
        15: [255, 255, 255],
        16: [192, 192, 192],
        17: [139, 0, 0],
        18: [0, 100, 0],
        19: [0, 0, 139],
        20: [255, 215, 0],
        21: [255, 140, 0],
        22: [139, 0, 139],
        23: [0, 206, 209],
        24: [75, 0, 130],
        25: [85, 107, 47],
        26: [34, 139, 34],
        27: [165, 42, 42],
        28: [70, 130, 180],
        29: [169, 169, 169],
        30: [32, 178, 170],
        31: [47, 79, 79],
        32: [245, 245, 245],
        33: [105, 105, 105],
        34: [205, 92, 92],
        35: [50, 205, 50],
        36: [65, 105, 225],
        37: [255, 223, 0],
        38: [255, 99, 71],
        39: [186, 85, 211],
        40: [0, 191, 255],
        41: [192, 192, 192],
    }


def _mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Convert an integer mask to an RGB image using the static color map."""
    color_map = _get_color_map()
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for label, rgb in color_map.items():
        color_img[mask == label] = rgb
    return color_img


# -----------------------------------------------------------------------------
# Core routine -----------------------------------------------------------------
# -----------------------------------------------------------------------------

def generate_eurocrops_mask(
    shp_path: str,
    bounding_box: tuple,
    mask_size: tuple = (10002, 10002),
    output_dir: str | None = None,
    force_create_output_dir: bool = True,
    save_png: bool = True,
    save_npy: bool = True,
):
    """Create a land‑cover mask for the given bounding box.

    Parameters
    ----------
    shp_path : str
        Absolute or relative path to the EuroCrops shapefile (\*.shp).
    bounding_box : tuple
        (east_min, east_max, north_min, north_max) in EPSG:2154.
    mask_size : tuple, optional
        (height, width) of the output mask.  Default matches the original
        script (10002, 10002).
    output_dir : str | None, optional
        Directory where the .npy and .png will be written.  If *None*, the
        files are **not** saved.
    force_create_output_dir : bool, optional
        Create *output_dir* if it doesn't exist.  Default ``True``.
    save_png / save_npy : bool, optional
        Control whether the coloured .png and/or raw .npy are written.

    Returns
    -------
    np.ndarray
        2‑D integer mask with the same logic as the original implementation.
    """
    east_min, east_max, north_min, north_max = bounding_box
    height, width = mask_size

    # ---------------------------------------------------------------------
    # Read & pre‑process the shapefile
    # ---------------------------------------------------------------------
    shp = gpd.read_file(shp_path, columns=["geometry", "EC_hcat_c"])
    shp = shp.dropna(subset=["EC_hcat_c"]).copy()
    shp["EC_hcat_c"] = shp["EC_hcat_c"].astype(int)

    # ---------------------------------------------------------------------
    # Build the transformer once.  Source CRS is fixed (2154 -> 32630).
    # ---------------------------------------------------------------------
    transformer = pyproj.Transformer.from_crs("EPSG:2154", "EPSG:32630", always_xy=True)

    mask = np.zeros((height, width), dtype=np.uint8)

    # Pre‑compute denominators for normalisation.
    span_east = east_max - east_min
    span_north = north_max - north_min

    # ---------------------------------------------------------------------
    # Iterate over polygons & paint the mask
    # ---------------------------------------------------------------------
    for _, row in shp.iterrows():
        hcat_code = row["EC_hcat_c"]
        if hcat_code not in ALL_CROPS_FLAT:
            continue  # Skip polygons we don't care about

        polygon = transform(transformer.transform, row.geometry)
        if polygon.geom_type != "Polygon":
            continue  # Skip multi‑geometries for simplicity

        coords = np.array(polygon.exterior.coords)
        # Translate to bounding‑box origin, then scale to pixel grid
        translated = coords - np.array([east_min, north_min])
        pixel_coords = (
            translated / np.array([span_east, span_north]) * np.array([width, height])
        ).astype(int)

        # Check if the polygon lies completely inside the bounding box
        if np.any(pixel_coords < 0) or np.any(pixel_coords[:, 0] >= width) or np.any(pixel_coords[:, 1] >= height):
            continue

        # Determine the class index (0‑based) from the ordered category list
        for label_index, (_, codes) in enumerate(CROP_CATEGORIES_ORDERED):
            if hcat_code in codes:
                cv2.fillPoly(mask, [pixel_coords], label_index)
                break

    # Flip vertically to match the original orientation
    mask = np.flipud(mask)

    # ---------------------------------------------------------------------
    # Optional output
    # ---------------------------------------------------------------------
    if output_dir is not None:
        if force_create_output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if save_npy:
            npy_path = os.path.join(output_dir, "eurocrops_mask.npy")
            np.save(npy_path, mask)
            print(f"Saved mask to {npy_path}")

        if save_png:
            png_path = os.path.join(output_dir, "eurocrops_mask.png")
            plt.figure(figsize=(10, 10))
            plt.imshow(_mask_to_color(mask))
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(png_path, bbox_inches="tight")
            plt.close()
            print(f"Saved coloured mask to {png_path}")

    return mask


# -----------------------------------------------------------------------------
# Example usage ---------------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    EXAMPLE_BBOX = (
        704_855.0, 804_875.0,  # east_min, east_max
        4_895_125.0, 4_995_145.0,  # north_min, north_max
    )

    SHAPEFILE = "../eurocrops/eurocrops_FR_2018/FR_2018/FR_2018_EC21.shp"
    OUTPUT_DIR = "./app/dataset/euro_mask"

    generate_eurocrops_mask(
        shp_path=SHAPEFILE,
        bounding_box=EXAMPLE_BBOX,
        mask_size=(10002, 10002),
        output_dir=OUTPUT_DIR,
        save_png=True,
        save_npy=True,
    )
