# ~/.tartvenv/bin/pip3 install minio tqdm
import os
import argparse
import datetime
import pytz

from tqdm import tqdm
from minio import Minio

MINIO_API_HOST = "s3.max.ac.nz"
BUCKET_NAME = "tart-hdf"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get data from the TART s3 bucket',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dir', required=False, default=".", help="Output directory")

    parser.add_argument('--target', required=False,
                        default='signal', help="Telescope name in s3 bucket.")

    parser.add_argument('--catalog', required=False,
                        default='https://tart.elec.ac.nz/catalog', help="Catalog API URL.")

    parser.add_argument('--n', required=False, type=int,
                        default=1, help="Number of HDF vis files.")
    parser.add_argument('--since', required=False, type=int,
                        default=60, help="Age in minutes.")

    ARGS = parser.parse_args()

    output = ARGS.dir
    if not os.path.exists(output):
        os.mkdir(output)


    # for o in objects:
    #     print(vars(o))
    #     print(dir(o))
    #     break

    N = ARGS.n
    minutes = ARGS.since

    utc_now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
    delta_t = datetime.timedelta(days = 0, hours=0, \
                    minutes=minutes,
                    seconds=0)

    an_hour_ago = utc_now  - delta_t

    # Do some logic to avoid boundaries and get the prefix as long as possible
    # This saves us from downloading the entire year's worth of objects just to
    # get some within the same day...
    prefix = f"rhodes/vis"
    if utc_now.year == an_hour_ago.year:
        prefix += f"/{utc_now.year}"
    if utc_now.month == an_hour_ago.month:
        prefix += f"/{utc_now.month}"
    if utc_now.day == an_hour_ago.day:
        prefix += f"/{utc_now.day}"
    print(f"Getting from path {prefix}")

    client = Minio(MINIO_API_HOST, secure=True)
    objects = client.list_objects(BUCKET_NAME, prefix=prefix, recursive=True, include_user_meta=True)



    desired_objects = []
    for o in objects:
        if o.last_modified > an_hour_ago:
            desired_objects.append(o)

    steps = len(desired_objects) // N

    index = 1
    for item in tqdm(list(desired_objects)[::steps]):

        dirname, fname = os.path.split(item.object_name)
        fname = f"obs_{index}.hdf"
        fname_out = os.path.join(output, fname)
        index = index+1
        client.fget_object(BUCKET_NAME, item.object_name, fname_out)
        print(fname_out)
