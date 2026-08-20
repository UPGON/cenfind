import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cenfind


def main():
    return cenfind.run(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
