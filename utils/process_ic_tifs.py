import rasterio
import numpy as np
import glob, os, argparse


def process_ic_tif(tif_path: str, out_dir: str):
    base = os.path.splitext(os.path.basename(tif_path))[0]  # e.g. S2A_30TYQATO_220503_IC
    with rasterio.open(tif_path) as src:
        # Most IC products are single-band; handle multi-band just in case
        for idx in range(1, src.count + 1):
            lai = src.read(idx).astype(np.float32)           # raw int16 → float32
            mask = lai <= 0                                  # nodata & all flags
            lai  = lai / 1000.0                              # scale to physical LAI
            lai[mask] = 0                                    # keep flags at 0

            out_name = f"{base}_measure_{idx-1:02d}.npy"
            np.save(os.path.join(out_dir, out_name), lai)
            print("✔ Created the LAI:  ", out_name)
