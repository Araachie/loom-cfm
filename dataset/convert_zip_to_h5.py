import argparse
import os
import sys

import numpy as np
from tqdm import tqdm
from zipfile import ZipFile
from PIL import Image

from h5 import HDF5Maker


def make_h5(zip_file, num_images, out_dir='./h5_ds', vids_per_shard=100000, force_h5=False):

    # H5 maker
    h5_maker = HDF5Maker(out_dir, num_per_shard=vids_per_shard, force=force_h5)

    with ZipFile(zip_file, "r") as zf:
        for image_index in tqdm(range(num_images), total=num_images):
            try:
                with zf.open(f"{image_index:05d}.png") as f:
                    image = np.array(Image.open(f).convert("RGB"), dtype=np.uint8)
                    h5_maker.add_data({"image": image}, dtype='uint8')
            except StopIteration:
                break
            except (KeyboardInterrupt, SystemExit):
                print("Ctrl+C!!")
                break
            except:
                e = sys.exc_info()[0]
                print("ERROR:", e)

    h5_maker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, help="Directory to save .hdf5 files")
    parser.add_argument('--zip_file', type=str, help="Zip file with images")
    parser.add_argument('--num_images', type=int, help="Number of images")
    parser.add_argument('--split', type=str, help="Data split")
    args = parser.parse_args()

    make_h5(
        zip_file=args.zip_file,
        out_dir=os.path.join(args.out_dir, args.split),
        num_images=args.num_images)
