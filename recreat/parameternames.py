from enum import Enum

class ParameterNames:

    class Reclassification(Enum):
        Mappings = 'mappings'
        ExportFilename = 'export_filename'
    
    class ClumpDetection(Enum):
        BarrierClasses = 'barrier_classes'
    
    class EdgeDetection(Enum):
        LandUseClasses = 'lu_classes'
        IgnoreEdgesToClass = 'ignore_edges_to_class'
        BufferEdges = 'buffer_edges'
    
    class ClassTotalSupply(Enum):
        Mode = 'mode'

    class AggregateClassTotalSupply(Enum):
        LandUseWeights = 'lu_weights'
        WriteNonWeightedResult = 'write_non_weighted_result'

    class AverageTotalSupplyAcrossCost(Enum):
        LandUseWeights = 'lu_weights'
        CostWeights = 'cost_weights'
        WriteNonWeightedResult = 'write_non_weighted_result'
        WriteScaledResult = 'write_scaled_result'

    class AverageDiversityAcrossCost(Enum):
        CostWeights = 'cost_weights'
        WriteNonWeightedResult = 'write_non_weighted_result'
        WriteScaledResult = 'write_scaled_result'

    class ComputeDistanceRasters(Enum):
        Mode = 'mode'
        LandUseClasses = 'lu_classes'
        AssessBuiltUp = 'assess_builtup' 


    