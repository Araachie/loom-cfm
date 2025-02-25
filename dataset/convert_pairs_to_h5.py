import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

from h5 import HDF5Maker


def get_pairs(root, num_pairs):
    for i in range(num_pairs):
        f = os.path.join(root, f"{i:09d}")
        source = np.load(os.path.join(f, "source.npy"))
        target = np.load(os.path.join(f, "target.npy"))

        yield {"source": source, "target": target}


def make_h5(data_root, num_pairs, out_dir='./h5_ds', vids_per_shard=100000, force_h5=False):

    # H5 maker
    h5_maker = HDF5Maker(out_dir, num_per_shard=vids_per_shard, force=force_h5)

    pairs_generator = get_pairs(data_root, num_pairs)

    for pair in tqdm(pairs_generator, total=num_pairs):
        try:
            h5_maker.add_data(pair, dtype='float32')
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
    parser.add_argument('--data_dir', type=str, help="Directory with pairs")
    parser.add_argument('--num_pairs', type=int, help="Number of pairs")
    args = parser.parse_args()

    make_h5(
        data_root=args.data_dir,
        out_dir=os.path.join(args.out_dir, 'train'),
        num_pairs=args.num_pairs)
