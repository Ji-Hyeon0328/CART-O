# import torch
# import torch.nn as nn

# class ObjectiveSelector(nn.Module):
#     def __init__(self, context_dim=256, cmd_dim=3, state_dim=36, output_dim=3):
#         """
#         Context와 명령을 바탕으로 보상 가중치를 결정하는 IRL 모듈
        
#         Args:
#             context_dim: c_t의 차원 (256)
#             cmd_dim: 주행 명령 차원 (3)
#             state_dim: 상태 추정값 차원 (36)
#             output_dim: 보상 가중치 개수 (beta_v, beta_s, beta_e)
#         """
#         super(ObjectiveSelector, self).__init__()
        
#         # 입력 차원 합계: 256 + 3 + 36 = 295
#         input_total_dim = context_dim + cmd_dim + state_dim
        
#         self.net = nn.Sequential(
#             nn.Linear(input_total_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, output_dim),
#             nn.Softmax(dim=-1) # 가중치의 합을 1로 제한하거나, 
#                                # 혹은 각 가중치의 범위를 제한하기 위해 Sigmoid/Softmax 사용
#         )

#     def forward(self, c_t, cmd, state):
#         """
#         Args:
#             c_t: Context Vector [Batch, 256]
#             cmd: Robot Commands [Batch, 3]
#             state: State Estimator/Proprioception [Batch, 36]
#         """
#         # 모든 입력을 하나로 결합
#         x = torch.cat([c_t, cmd, state], dim=-1)
        
#         # 보상 가중치 R (beta 값들) 출력
#         reward_weights = self.net(x)
#         return reward_weights

import torch
import torch.nn as nn

class ObjectiveSelector(nn.Module):
    def __init__(
        self,
        context_dim=256,
        cmd_dim=3,
        state_dim=36,
        aux_dim=5,          # 예: friction 4 + base_height 1 + slip 4
        output_dim=3,
        temperature=0.1,
        min_weight=0.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.min_weight = min_weight

        input_total_dim = context_dim + cmd_dim + state_dim + aux_dim

        self.backbone = nn.Sequential(
            nn.Linear(input_total_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.head = nn.Linear(128, output_dim)

    def forward(self, c_t, cmd, state, aux_state):
        x = torch.cat([c_t, cmd, state, aux_state], dim=-1)
        h = self.backbone(x)
        logits = self.head(h) / self.temperature
        w = torch.softmax(logits, dim=-1)

        eps = self.min_weight
        w = (1.0 - 3.0 * eps) * w + eps
        return w