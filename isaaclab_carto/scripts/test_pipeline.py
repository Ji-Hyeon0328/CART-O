import os
import sys
import torch

# 1. 경로 설정 (반드시 임포트보다 먼저 실행되어야 합니다!)
# 현재 파일(scripts) -> isaaclab_carto -> isaaclab_carto -> source 폴더까지 3단계 상승
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))

if SOURCE_DIR not in sys.path:
    sys.path.append(SOURCE_DIR)

# 2. 이제 CARTO 모듈들을 임포트합니다.
try:
    from isaaclab_carto.isaaclab_carto.utils.history_buffer import HistoryBuffer
    from isaaclab_carto.isaaclab_carto.networks.encoders import VisualEncoder, SurfaceEncoder, ProprioEncoder
    from isaaclab_carto.isaaclab_carto.networks.attention import CARTOAttention
    from isaaclab_carto.isaaclab_carto.networks.objective_selector import ObjectiveSelector
    from isaaclab_carto.isaaclab_carto.networks.tss import TSSModule
    from isaaclab_carto.isaaclab_carto.networks.policy import CARTOActionPolicy
    print("모듈 임포트 성공!")
except ImportError as e:
    print(f"임포트 실패: {e}")
    print(f"현재 탐색 경로(sys.path): {sys.path}")
    sys.exit(1)

def run_carto_pipeline_test():
    # --- 1. 환경 설정 파라미터 ---
    num_envs = 4        # 가상 Batch Size
    horizon = 10        # 과거 이력 길이
    proprio_dim = 36    # 고유 수용 감각 차원
    feature_dim = 256   # 인코더 출력 차원
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- CARTO 파이프라인 테스트 시작 (Device: {device}) ---")

    # --- 2. 모듈 초기화 ---
    buffer = HistoryBuffer(num_envs, horizon, proprio_dim, device)
    
    vis_enc = VisualEncoder(feature_dim).to(device)
    sur_enc = SurfaceEncoder(input_dim=128, feature_dim=feature_dim).to(device)
    pro_enc = ProprioEncoder(input_dim=proprio_dim, hidden_dim=128, feature_dim=feature_dim).to(device)
    
    attention = CARTOAttention(cmd_dim=3, h_t_dim=256, encoder_feature_dim=feature_dim).to(device)
    obj_selector = ObjectiveSelector(context_dim=256, cmd_dim=3, state_dim=36).to(device)

    # --- 3. 가상 데이터 생성 (Observation 시뮬레이션) ---
    # 실제 mdp/observations.py에서 나올 데이터들의 형태
    mock_rgbd = torch.randn(num_envs, 4, 64, 64).to(device)    # Sv용
    mock_mesh = torch.randn(num_envs, 128).to(device)          # Sm용
    mock_proprio = torch.randn(num_envs, proprio_dim).to(device) # Sp용 (현재 스텝)
    mock_cmd = torch.tensor([[1.0, 0.0, 0.0]] * num_envs).to(device) # Cmd (전진 명령)

    # --- 4. 데이터 흐름 테스트 ---
    print("\n[Step 1] Buffer Update")
    buffer.update(mock_proprio)
    history = buffer.get_history()
    print(f"-> History Buffer Shape: {history.shape}") # [4, 10, 36]

    print("\n[Step 2] Encoding Features")
    s_v = vis_enc(mock_rgbd)
    s_m = sur_enc(mock_mesh)
    s_p, h_t = pro_enc(history) # h_t는 BiLSTM의 hidden state 결합본
    
    print(f"-> s_v Shape: {s_v.shape}") # [4, 256]
    print(f"-> s_m Shape: {s_m.shape}") # [4, 256]
    print(f"-> s_p Shape: {s_p.shape}") # [4, 256]
    print(f"-> h_t (Query) Shape: {h_t.shape}") # [4, 256]

    print("\n[Step 3] Attention Fusion")
    # 인코더 특징들을 하나로 묶음
    features = torch.stack([s_v, s_m, s_p], dim=1) # [4, 3, 256]
    
    # CARTO 전용 Attention 실행 (Cmd + h_t 사용)
    c_t, weights = attention(mock_cmd, h_t, features)
    
    print(f"-> Attention Weights (Sv, Sm, Sp): \n{weights}")
    print(f"-> 최종 Context Vector (c_t) Shape: {c_t.shape}") # [4, 256]

    print("\n--- 모든 모듈이 정상적으로 연결되었습니다! ---")

    print("\n[Step 4] Objective Selection (IRL)")
    # s_p를 현재 state로 가정하여 입력
    reward_params = obj_selector(c_t, mock_cmd, mock_proprio)

    print(f"-> 최종 보상 가중치 (beta_v, beta_s, beta_e): \n{reward_params}")
    print(f"-> 가중치 형태: {reward_params.shape}") # [4, 3]

    # 1. TSS 초기화 및 가상 라이브러리 생성
    tss = TSSModule().to(device)
    mock_library = torch.randn(20, 512).to(device) # 20개의 과거 경험 시퀀스

    # 2. TSS 실행
    best_idx, all_scores = tss(mock_library, s_p, c_t)

    print("\n[Step 5] Temporal Sequence Selection (TSS)")
    print(f"-> 선택된 최적 경험 인덱스: {best_idx}")
    print(f"-> 시퀀스 스코어 분포 (평균): {all_scores.mean().item():.4f}")

    # 1. Policy 초기화
    policy = CARTOActionPolicy().to(device)

    # 2. 최종 Action 생성
    # s_p(현재 상태)와 c_t(Attention 결과) 사용
    final_action = policy(mock_proprio, c_t)

    print("\n[Step 6] Final RL Policy Output")
    print(f"-> Spot 실행 명령 (v_x, v_y, yaw, height): \n{final_action}")
    print(f"-> 명령 형태: {final_action.shape}") # [4, 4] (Batch, Actions)

    print("\n--- CARTO High-level Planner 전체 아키텍처 검증 완료! ---")

if __name__ == "__main__":
    run_carto_pipeline_test()