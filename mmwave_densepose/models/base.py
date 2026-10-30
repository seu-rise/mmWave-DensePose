from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch.nn as nn

class BaseModel(nn.Module, ABC):

    def __init__(self, info: Dict[str, Any]):
        super().__init__()
        self.info = info

    @abstractmethod
    def forward(self, x):
        pass

    @property
    @abstractmethod
    def task_type(self) -> str:
        pass

class KeypointModel(BaseModel):

    @property
    def task_type(self) -> str:
        return 'keypoint'

    @property
    def num_classes(self) -> Optional[int]:
        return self.info.get('num_classes', None)

    @property
    def num_keypoints(self) -> int:
        return self.info.get('num_keypoints', 25)

    @property
    def max_points(self) -> int:
        return self.info.get('max_points', 300)

class DenseModel(BaseModel):

    @property
    def task_type(self) -> str:
        return 'dense'

    @property
    def num_in_points(self) -> int:
        return self.info.get('num_in_points', 300)

    @property
    def num_out_points(self) -> int:
        return self.info.get('num_out_points', 1200)
