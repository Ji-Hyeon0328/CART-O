# High-Level Framework Checkpoint Summary

## 목적
- Option 2 통합 high-level planner로 넘어가기 전의 checkpoint를 남긴다.
- Objective Selector(beta-dependent reward)와 TSS의 현재 성능을 기록한다.

## 주요 아티팩트
- **beta_analysis_pairs_csv**: `/home/shoko/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/beta_analysis/beta_analysis_pairs.csv`
- **tss_ckpt**: `/home/shoko/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/tss_checkpoints/spot_tss_best.pt`
- **tss_cluster_centers_csv**: `/home/shoko/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/tss_labels/tss_cluster_centers.csv`
- **tss_cluster_summary_csv**: `/home/shoko/IsaacLab/source/isaaclab_carto/data/processed/jy_run1_ros2/tss_labels/tss_cluster_summary.csv`

## Beta Analysis
- Num pairs: `1000`
- Better beta mean: `{'better_beta_v': 0.30568317212164403, 'better_beta_h': 0.2657512218505144, 'better_beta_e': 0.4285656041800976}`
- Better beta std: `{'better_beta_v': 0.017841182227464242, 'better_beta_h': 0.014521600044528576, 'better_beta_e': 0.032194864783524735}`
- Worse beta mean: `{'worse_beta_v': 0.2578149066418409, 'worse_beta_h': 0.22661034394800664, 'worse_beta_e': 0.5155747523903846}`
- Worse beta std: `{'worse_beta_v': 0.028913105938369592, 'worse_beta_h': 0.02346844006561114, 'worse_beta_e': 0.05226350708685796}`

### 해석
- beta는 uniform에서 벗어나기 시작했다.
- worse 샘플에서 energy 쪽 beta 비중이 더 커지는 경향이 관찰되었다.
- Objective Selector가 reward ranking뿐 아니라 objective preference 분화도 일부 학습하기 시작한 상태로 해석할 수 있다.

## TSS Training
- Best epoch: `10`
- Train stats: `{'loss': 0.5477625661719162, 'acc': 0.7743362831858407}`
- Val stats: `{'loss': 0.561281321140436, 'acc': 0.7596153846153846}`

## TSS Clustering
- Cluster centers: `[{'center_beta_v': 0.3193837319345821, 'center_beta_h': 0.2768134496408028, 'center_beta_e': 0.4038028165441626, 'cluster_id': 0}, {'center_beta_v': 0.2906613843183597, 'center_beta_h': 0.2536221964116866, 'center_beta_e': 0.4557164174580724, 'cluster_id': 1}]`
- Cluster summary: `[{'cluster_id': 0, 'count': 523, 'beta_v_mean': 0.3193837319345821, 'beta_h_mean': 0.2768134496408028, 'beta_e_mean': 0.4038028165441626, 'beta_v_std': 0.0093547101247215, 'beta_h_std': 0.0081042151322093, 'beta_e_std': 0.0171369372754371, 'J_v_mean': 0.2942257303509498, 'J_h_mean': -2.7654833064252515, 'J_e_mean': -57.419966148828685, 'reward_mean': -23.83005750885885, 'margin_mean': 9.503185144573964}, {'cluster_id': 1, 'count': 477, 'beta_v_mean': 0.2906613843183597, 'beta_h_mean': 0.2536221964116866, 'beta_e_mean': 0.4557164174580724, 'beta_v_std': 0.0118199055153041, 'beta_h_std': 0.0094165814092822, 'beta_e_std': 0.0209964400019921, 'J_v_mean': 0.2777675670967974, 'J_h_mean': -2.757054926214478, 'J_e_mean': -53.38307077979642, 'reward_mean': -24.926419180144304, 'margin_mean': 8.508594884812457}]`

### 해석
- k=2 clustering 기준으로 objective preference가 두 개의 pseudo mode로 분리되었다.
- 현재 mode는 gait primitive라기보다는 aggressive/conservative behavior tendency에 가까운 해석이 적절하다.

## 다음 단계 (Option 2)
- shared encoder/context 위에 beta head와 z head를 함께 두는 통합 high-level planner로 이동
- loss는 preference loss + z classification loss를 joint하게 사용
- 초기화는 현재 preference/TSS checkpoint를 활용
