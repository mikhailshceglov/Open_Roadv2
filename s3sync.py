"""Move data between the job container and Object Storage.

DataSphere's own S3 mounts need a connector that can only be created in the
project UI -- there is no API for it (the v2 endpoints 404). Rather than block
on a manual step, the container talks to Object Storage directly with a service
account's static key, which is plain S3 and needs nothing configured in the
project.

Credentials come from AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in the
environment; they are never written to the repository.
"""

import argparse
import os
from pathlib import Path
import sys

import boto3
from botocore.config import Config

ENDPOINT = "https://storage.yandexcloud.net"
REGION = "ru-central1"


def client():
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if not os.environ.get(name):
            raise SystemExit("{} is not set; cannot reach Object Storage".format(name))
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        region_name=REGION,
        # This network drops connections often enough that the default of one
        # attempt turns a working pipeline into a flaky one.
        config=Config(retries={"max_attempts": 8, "mode": "standard"}),
    )


def get(bucket: str, key: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print("s3: get {} -> {}".format(key, destination), flush=True)
    client().download_file(bucket, key, str(destination))
    print("s3: {} ({:.1f} MB)".format(destination.name, destination.stat().st_size / 2**20), flush=True)


def put_dir(bucket: str, source: Path, prefix: str) -> None:
    if not source.is_dir():
        print("s3: nothing to upload, {} does not exist".format(source), flush=True)
        return
    s3 = client()
    count = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        key = "{}/{}".format(prefix.rstrip("/"), path.relative_to(source))
        s3.upload_file(str(path), bucket, key)
        count += 1
    print("s3: uploaded {} files to {}".format(count, prefix), flush=True)


def put(bucket: str, source: Path, key: str) -> None:
    if not source.is_file():
        print("s3: nothing to upload, {} does not exist".format(source), flush=True)
        return
    client().upload_file(str(source), bucket, key)
    print("s3: put {} -> {}".format(source, key), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("get", "put", "put-dir"))
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET"))
    args = parser.parse_args()

    if not args.bucket:
        raise SystemExit("S3_BUCKET is not set")

    if args.action == "get":
        get(args.bucket, args.source, Path(args.destination))
    elif args.action == "put":
        put(args.bucket, Path(args.source), args.destination)
    else:
        put_dir(args.bucket, Path(args.source), args.destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
