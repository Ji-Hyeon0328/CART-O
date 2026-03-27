import torch
import torch.nn as nn

# 1. 파일 로드
path = "logs/rsl_rl/spot_flat/2026-03-06_21-32-35/model_0.pt"
try:
    checkpoint = torch.load(path, map_location="cpu")
    print("✅ 파일 로드 성공!")
except Exception as e:
    print(f"❌ 로드 실패: {e}")
    exit()

# 2. 내부 키(Key) 확인
if 'model_state_dict' in checkpoint:
    weights = checkpoint['model_state_dict']
else:
    weights = checkpoint

# 3. 입력 차원(Input Dimension) 확인
# actor.0.weight의 shape는 [출력 크기, 입력 크기]입니다.
if 'actor.0.weight' in weights:
    layer_weight = weights['actor.0.weight']
    hidden_dim, input_dim = layer_weight.shape
    print(f"\n--- 모델 분석 결과 ---")
    print(f"📍 기대하는 입력 차원(Obs): {input_dim}")
    print(f"📍 첫 번째 은닉층 크기: {hidden_dim}")
    
    # 출력 차원 확인 (마지막 레이어)
    last_key = 'actor.6.bias' # 4층 신경망 기준
    if last_key in weights:
        output_dim = weights[last_key].shape[0]
        print(f"📍 출력 차원(Actions): {output_dim}")
else:
    print("❌ 'actor.0.weight' 키를 찾을 수 없습니다. 모델 구조를 다시 확인해야 합니다.")
    print("사용 가능한 키 목록:", weights.keys())
    exit()

# 4. 가상 테스트 (Forward Pass Test)
class TestActor(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x): return self.net(x)

try:
    test_model = TestActor(input_dim, output_dim)
    # 'actor.' 접두어 제거 후 로드
    clean_weights = {k.replace('actor.', 'net.'): v for k, v in weights.items() if 'actor' in k}
    test_model.load_state_dict(clean_weights, strict=False)
    
    # 가상의 입력값(48D 또는 확인된 input_dim)을 넣어봅니다.
    dummy_input = torch.randn(1, input_dim)
    output = test_model(dummy_input)
    print(f"\n✅ 가상 추론 성공! 출력값 형태: {output.shape}")
    print("👉 이 모델은 정상이며, 입력 차원만 맞추면 작동합니다.")
except Exception as e:
    print(f"\n❌ 추론 테스트 실패: {e}")