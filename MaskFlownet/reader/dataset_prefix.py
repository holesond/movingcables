import os

data_prefix = None

if os.path.isdir('/nfs/datasets'):
    data_prefix = r'/nfs/datasets'
elif os.path.isdir('/home/holesond/datasets'):
    data_prefix = r'/home/holesond/datasets'
else:
    raise RuntimeError("MaskFlownet reader could not find any "
        "dataset folder.")
    
#data_prefix = r'/home/holesond/dataset_mnt'
#data_prefix = r'/nfs/datasets'
#data_prefix = r'/home/holesond/datasets'
