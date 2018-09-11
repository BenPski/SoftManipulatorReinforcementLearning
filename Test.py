"""
recording the performance of a learned policy

read in a config and run the relevant tests on it
"""

from config_utils import ConfigHandler
import argparse
 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing a learned policy")
    parser.add_argument('config', metavar='config',type=str,help='The config that stores a tests data')

    args = parser.parse_args()

    test = ConfigHandler.fromFile(args.config)

    test.run_test(save=True)
