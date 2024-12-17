import logging
import sys

from openff.qcsubmit.results import OptimizationResultCollection

from yammbs.inputs import QCArchiveDataset

logging.getLogger("openff").setLevel(logging.ERROR)


def main():
    print("loading collection", file=sys.stderr)
    opt = OptimizationResultCollection.parse_raw(sys.stdin.read())
    print("caching records", file=sys.stderr)
    crc = QCArchiveDataset.from_qcsubmit_collection(opt)
    print(crc.model_dump_json())


if __name__ == "__main__":
    main()
