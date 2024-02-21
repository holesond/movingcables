from . import layer
from . import MaskFlownet
from . import MaskFlownetProb
from . import pipeline
from . import pipeline_prob

def get_pipeline(network, **kwargs):
    if network == 'MaskFlownet':
        return pipeline.PipelineFlownet(**kwargs)
    elif network == 'MaskFlownetProb':
        return pipeline_prob.PipelineFlownetProb(**kwargs)
    else:
        raise NotImplementedError
