# argparse
import argparse as arg
import numpy as np


#from recreat import recreat

def recreat_util():

    parser = arg.ArgumentParser()

    # working directory
    parser.add_argument('-w', '--working-directory', help='Set working directory', dest='wd', metavar='path', action='store', type=str, required=True)
    # verbose
    parser.add_argument('-v', '--verbose', help='Enable verbose reporting', action='store_true')
    # dtype definition
    parser.add_argument('--datatype', help='Define project datatype.', metavar='datatype', action='store', nargs='?', choices=['int', 'float', 'double'], type=str)


    # root directory
    parser.add_argument('-r', '--root', help='Specify data root of the scenario to be assessed', metavar='root folder', required=True, type=str)
    # land-use map
    parser.add_argument('--lu-map', help='Specify scenario-specific land-use map', metavar='filename.tif', dest='lu_filename', required=True, type=str)

    # aggregations: this is a bit ugly as-is
    parser.add_argument('--ad', nargs=1, action='append', type=int)
    parser.add_argument('--ao', nargs='*', action='append', type=int)

    # define classes
    parser.add_argument('-p', '--patches', help='Define patch recreational classes', dest='patches', metavar='classes', action='extend', nargs='*', type=int)
    parser.add_argument('-e', '--edges', help='Define edge recreational classes', dest='edges', metavar='classes', action='extend', nargs='*', type=int)
    parser.add_argument('-b', '--built-up', help='Define edge recreational classes', dest='builtup', metavar='classes', action='extend', nargs='*', type=int)

    # define costs
    parser.add_argument('-c', '--costs', help='Define cost thresholds (kernel size) in pixel units. Kernel size must be an odd number.', metavar='costs', action='extend', nargs='*', type=int)

    # variables for methods
    parser.add_argument('--clump-barriers', help='Define classes acting as barriers separating clumps.', metavar='classes', dest='clump_barriers', nargs='*', action='extend', type=int)
    parser.add_argument('--ignore-edge-to-class', help='Define class to which edges will be ignored.', metavar='class', dest='edge_barrier', action='store', type=int)
    parser.add_argument('--filter-method', help='Define method used to conduct moving window operations.', metavar='method-name', dest='filter_method', choices=['generic_filter', 'convolve', 'ocv_filter2d'], action='store', type=str)
    parser.add_argument('--proximity-method', help='Define method used to determine proximity rasters.', metavar='method-name', dest='proximity_method', choices=['xr', 'dr'], action='store', type=str)
    parser.add_argument('--proximity-to-builtup', help='Compute proximities to built-up.', dest='proximity_builtup', action='store_true')


    # operations to do
    parser.add_argument('-a', '--clumps', dest='clumps', help='Run clump detection.', action='store_true')
    parser.add_argument('-m', '--masking', dest='masking', help='Run land-use masking.', action='store_true')
    parser.add_argument('-l', '--detect-edges', dest='edge_detection', help='Run edge detection.', action='store_true')
    parser.add_argument('-s', '--class-supply', dest='class_supply', help='Run class total supply estimation.', action='store_true')
    parser.add_argument('-d', '--class-diversity', dest='class_diversity', help='Run class diversity estimation.', action='store_true')
    parser.add_argument('-x', '--proximities', dest='proximities', help='Run proximity raster estimation.', action='store_true')




    args = parser.parse_args()

    # process some further defaults
    clump_barriers = [0] if args.clump_barriers is None else args.clump_barriers
    filter_method = 'ocv_filter2d' if args.filter_method is None else args.filter_method
    proximity_method = 'xr' if args.proximity_method is None else args.proximity_method

    # instantiate class
    #recreat_model = recreat(args.wd)
    #recreat_model.set_params('verbose-reporting', args.verbose)

    # define dtype if provided
    if args.datatype is not None:
        # map cli option to np dtype
        if args.datatype == 'int32':
            target_type = np.int32
        elif args.datatype == 'float':
            target_type = np.float32
        elif args.datatype == 'double':
            target_type = np.double

        #recreat_model.set_params('use-data-type', target_type)


    # import land-use map from scenario root
    #recreat_model.set_land_use_map(args.root, args.lu_filename)

    # conduct aggregations as needed
    if args.ad is not None:
        class_aggregations = {}
        for i in range(len(args.ad)):
            class_aggregations[args.ad[i][0]] = args.ao[i]

        print('run aggregation')
        #recreat_model.aggregate_classes(aggregations=class_aggregations)            

    # specify classes
    #recreat_model.set_params('classes.patch', args.patches)
    #recreat_model.set_params('classes.edge', args.edges)
    #recreat_model.set_params('classes.builtup', args.builtup)
    #recreat_model.set_params('costs', args.costs) 

    # done with model specifications
    # conduct operations as requested by user
    if args.clumps:
        print('run clumps')
        #recreat_model.detect_clumps(barrier_classes=clump_barriers)
    if args.masking:
        print('run masking')
        #recreat_model.mask_landuses()    
    if args.edge_detection:
        print('run edge detection')
        # note that args.edge_barrier can be int or None
        #recreat_model.detect_edges(ignore_edges_to_class=args.edge_barrier)
    if args.class_supply:
        print('run class total supply')
        #recreat_model.class_total_supply(mode = filter_method)
    if args.class_diversity:
        print('run class diversity')
        #recreat_model.class_diversity()
    if args.proximities:
        print('run proximities')
        #recreat_model.compute_distance_rasters(mode=proximity_method, assess_builtup=args.proximity_builtup)  


recreat_util()