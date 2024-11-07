# argparse
import argparse as arg
import ast
from recreat import recreat

def recreat_util():

    parser = arg.ArgumentParser()

    # working directory
    parser.add_argument('--wd', help='Set working directory', metavar='path', action='store', type=str, required=True)
    # root directory
    parser.add_argument('--root', help='Specify data root of the scenario to be assessed', metavar='root folder', required=True, type=str)
    # land-use map
    parser.add_argument('--lu-map', help='Specify scenario-specific land-use map', metavar='filename.tif', dest='lumap', required=True, type=str)


    parser.add_argument('-a', '--aggregate', action='append', type=str)

    # patch classes
    parser.add_argument('-p', '--patches', help='Define patch recreational classes', metavar='classes', action='extend', nargs='*', type=int)
    # edge classes
    parser.add_argument('-e', '--edges', help='Define edge recreational classes', metavar='classes', action='extend', nargs='*', type=int)


    args = parser.parse_args()

    # working directory
    working_dir = args.wd
    root_dir = args.root    
    lu_filename = args.lumap

    for dict_as_str in args.aggregate:
        print(list(ast.literal_eval(dict_as_str.split('=')[1])))

    print(args.patches)

    #recreat_model = recreat(working_dir)

recreat_util()