import sys
import os
import cenfind

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main():
    return cenfind.run(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
