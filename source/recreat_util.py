# argparse
import argparse as arg


from .recreat import recreat

parser = arg.ArgumentParser()

# working directory
parser.add_argument("--wd", help="Set working directory", action="store", type=str, required=True)
# land-use map
parser.add_argument("--root", help="Specify data root of scenario", required=True, type=str)
parser.add_argument("--lu", help="Specify land-use map", required=True, type=str)


# patch classes
parser.add_argument("-p", "--patches", help="Define patch recreational classes", action="extend", nargs="*", type=int)
# edge classes
parser.add_argument("-e", "--edges", help="Define edge recreational classes", action="extend", nargs="*", type=int)


args = parser.parse_args()

print(args.wd)
print(args.root)
print(args.lu)
print(args.patches)
