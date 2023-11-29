# ~/.tartvenv/bin/pip3 install minio tqdm
import os
import argparse
import datetime
import pytz

from tqdm import tqdm
from minio import Minio

MINIO_API_HOST = "s3.max.ac.nz"
BUCKET_NAME = "tart-hdf"

# prefix = 'rhodes/raw/2022/'
# prefix = 'rhodes/vis/2022/'
# prefix = 'signal/raw/2022/'
prefix = 'rhodes/vis/2023/'

output = 'downloads/'

limit_last_10 = -10     # Only fetch the last 10 entries files


if __name__ == "__main__":
    if not os.path.exists(output):
        os.mkdir(output)

    client = Minio(MINIO_API_HOST, secure=True)
    objects = client.list_objects(BUCKET_NAME, prefix=prefix, recursive=True, include_user_meta=True)

    # for o in objects:
    #     print(vars(o))
    #     print(dir(o))
    #     break

    N = 4
    minutes = 90

    utc_now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)

    prefix = f"rhodes/vis/{utc_now.year}/{utc_now.month}"

    delta_t = datetime.timedelta(days = 0, hours=0, \
                    minutes=minutes,
                    seconds=0)

    an_hour_ago = utc_now  - delta_t


    last_hour = []
    for o in objects:
        if o.last_modified > an_hour_ago:
            last_hour.append(o)

    steps = len(last_hour) // N

    for item in tqdm(list(last_hour)[::steps]):
        fname_out = f'{output}{item.object_name}'
        client.fget_object(BUCKET_NAME, item.object_name, fname_out)
        print(fname_out)
