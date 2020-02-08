# Hashing-based Non-Maximum Suppression for Crowded Object Detection

## Installation
The project is based on [facebook/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
Please refer to it for installation.

## Hashing-based Non-Maximum Suppression
Here are example codes
```python
import torch
from maskrcnn_benchmark.layers import MultiHashNMSAnyKPt
hnms = MultiHashNMSAnyKPt(num=1, alpha=0.7)
rects = [[10, 20, 10, 20], [10, 20, 10, 20], [30, 6, 4, 5]]
conf = [0.9, 0.8, 0.9]
rects = torch.tensor(rects).float()
conf = torch.tensor(conf)
keep = hnms(rects, conf)
print(keep)
# the output is
# tensor([2, 0])
```
