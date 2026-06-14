# aerial_box_transport

듀얼암 공중 매니퓰레이터(쿼드로터 드론 + 양팔 그리퍼)가 박스를 책상에서 집어 랙 선반에
옮기는 **접촉 인지 전신 궤적 최적화(whole-body OCP)** 플래너입니다. 핵심 아이디어는 운반
가속도에 맞춰 그리퍼 스퀴즈 힘을 조절하는 **미끄럼 인지(slip-aware) 힘 규제**이고, 생성된
궤적을 IsaacSim에서 재생해 검증합니다.

설계 결정(D1-D8)과 빌드 계획은 repo 루트의 `../IMPLEMENTATION_SPEC.md`,
`../ROADMAP.md`에 있습니다. 플래너 절반(`src/model`, `src/planner`)은 순수 Python
(Pinocchio + CasADi + IPOPT)이라 IsaacSim 없이 실행/단위테스트가 됩니다.

## 무엇을 하는가
박스의 집는 위치(pick)와 놓는 위치(place)만 주면, 드론이 집(spawn)에서 출발해 다음을
수행하는 전신 궤적을 한 번에 생성합니다. 핵심은 **경로를 미리 주지 않는다**는 것입니다.
비행 경로, 팔이 내려가는 동작, 박스의 운반 경로 어느 것도 reference로 추종하지 않고,
제약과 비용으로부터 최적화가 스스로 발견합니다(가이드 없음, IPOPT 초기 seed만 둠).

- **접근**: 팔을 수평으로 편 채 출발해 박스 바로 위로 비행한 뒤 수직으로 하강. 이 비행
  경로는 정렬 비용(아래)이 base를 박스 위로 끌고 파지 제약이 아래로 끌어 나온 결과입니다.
  닫기 전 그리퍼를 넉넉히(약 30 cm) 벌려 작은 박스를 패드가 건드리지 않게 합니다.
- **파지**: 두 그리퍼 패드가 있어야 할 위치(박스 좌/우 면의 중심)만 지정하면, whole-body
  planning이 그 두 표적에 도달하는 **base와 팔 궤적을 스스로 찾습니다**(미리 계산한 자세를
  추종하지 않음). base는 책상 표면 위로 일정 거리 떠 있고(지상효과/충돌 회피), 팔만 대각선
  아래로 내려가 박스에 닿습니다. 두 패드는 **평행하게** 박스 양면을 누르고(좌우 팔은 거울
  대칭), 가속도에 맞춘 스퀴즈 힘으로 **마찰 파지**(미끄럼 방지)합니다.
- **운반**: 박스 중심이 base 중심 바로 아래 오도록 하는 **정렬 비용**으로 운반 중 외란
  토크와 무게중심 이동을 최소화합니다(omnidirectional 드론). 박스 중심을 집는/놓는 위치
  위의 좁은 수직 원기둥에 가두고 책상/랙을 바운딩 박스로 회피해, 위로 올라갔다 수직으로
  내려오는 경로를 최적화가 **스스로 발견**합니다.
- **놓기**: 선반 위에서 팔은 그대로 둔 채 그리퍼만 벌려 박스를 내려놓습니다.

팔이 아래로 내려가는 동작도 명시적으로 지정한 것이 아니라, "base는 표면에서 일정 거리
이상 떨어져 있어야 한다"는 제약과 "박스 좌/우 면 중심에 패드가 닿아야 한다"는 조건으로부터
최적화가 스스로 찾아낸 결과입니다.

## 디렉토리 구조
- `config/robot.yaml`  접촉 강성 k, 평활화 eps, 마찰계수 mu, 스퀴즈 마진
- `config/task.yaml`   박스 질량/크기, pick/place, 장애물(책상/랙) 바운딩 박스, 팔 설정
  (arm_grasp/arm_pregrasp, dof1 포함), base_clearance, ee_radius, 단계별 시간, 가속도 스윕
- `config/usd_model.json`, `config/isaac_model.json`  USD/PhysX에서 추출한 운동학/관성
- `src/model/`     전신 모델(`whole_body.py`, USD->Pinocchio), 평활 접촉(`contact.py`),
  박스/마찰콘(`box.py`), USD/그래스프 분석 도구
- `src/planner/`   다단계 OCP(`ocp.py`), 미끄럼 인지 힘 법칙(`slip_aware.py`),
  시간 이산화(`transcription.py`), OCP 문서 생성(`make_ocp_pdf.py`).
  플래너 비교 연구: 샘플링 플래너 sweep(`sampling_compare.py`), grasp-constrained 운반
  CBiRRT(`grasp_constrained.py`), 샘플 경로->트래커 변환(`sampling_to_reference.py`),
  샘플->OCP 하이브리드(`hybrid_seed_ocp.py`), 비교 그림(`plot_compare.py`), OCP 시간측정
  (`time_ocp.py`).
  **창문 통과 연구**: 창문 keep-out을 추가한 OCP(`ocp.py`의 `solve_ocp(window=True)`),
  창문 충돌을 coal(GJK/EPA)로 보는 샘플러(`sampling_compare.py`), 윈도우 CBiRRT 테스트
  (`window_test.py`), 샘플->OCP 윈도우 하이브리드(`hybrid_window.py`).
  **키프레임 가이드**(메인 연구): 제약 일관 보간 seed 생성(`build_kf_seed.py`), 키프레임
  유도 OCP(`keyframe_window.py`), 세 레퍼런스의 coal 정량 검증(`verify_window.py`).
  시각화/문서: 재생 파일 변환(`export_play.py`), 한국어 방법론 PDF(`make_window_method_pdf.py`),
  안정 동작 버전 백업(`ocp_soft_working.py`)
- `src/sim/`       gRITE SE(3) 컨트롤러(`grite_controller.py`), IsaacSim 동역학 재생/검증
  (`track_reference.py`, 동적 박스 + 실제 창 충돌체), 계획 경로의 기구학적 재생
  (`play_path.py`, 계획 자세를 그대로 포즈; `--proxy`로 창문 keep-out 볼륨 표시)
- `src/baselines/`, `src/experiments/`  고정력 베이스라인, 가속도 스윕/헤드라인 플롯
- `tests/`         플래너 단위테스트(pytest)
- `docs/`          `OCP_formulation.pdf`(비용 함수 + 제약조건, 한국어),
  `window_method_ko.pdf`(창문 통과 방법론, 한국어), `window_keyframe_results.md`
  (키프레임 가이드 3종 비교 결과 + 재현 명령)
- `results/`       생성 산출물(궤적 npz, 그림). git에는 포함되지 않음(재생성 가능).
  창문 레퍼런스: `window_reference_soft_box.npz`(샘플러 가이드), `window_reference_wedge_only.npz`
  (wedge 제약), `window_reference_keyframe.npz`(키프레임 가이드)

## 환경 (conda 3개)
환경 자체(설치된 패키지 폴더)는 git에 올리지 않습니다(수 GB, 플랫폼 의존 바이너리).
아래처럼 명세로부터 재생성하세요.

- **`am_dualarm`** (플래너): Pinocchio(+pinocchio.casadi/cpin), CasADi, IPOPT, numpy,
  scipy, matplotlib, pyyaml, pytest, usd-core. OCP를 풀어 레퍼런스 궤적을 생성.
  `environment.yml`로 재생성:
  ```
  conda env create -f environment.yml
  conda activate am_dualarm
  ```
- **`am_isaac`** (시뮬레이션): IsaacSim 5.1.0 / IsaacLab 0.54.3. NVIDIA 공식 절차로
  **별도 설치**합니다(pip wheel 기반, 수 GB라 environment.yml로 재생성하지 않음). 이
  IsaacSim 씬을 만든 환경과 동일한 IsaacSim/IsaacLab을 쓰면 됩니다. 설치는 NVIDIA
  Isaac Sim / Isaac Lab 공식 문서를 참고하세요.
- **`am_sampling`** (샘플링 비교): OMPL + Pinocchio. 샘플링 기반 플래너 비교용. 재생성:
  ```
  conda create -n am_sampling -c conda-forge python=3.11 pinocchio numpy scipy pyyaml matplotlib
  conda run -n am_sampling pip install ompl casadi
  ```
  (conda-forge의 ompl에는 python 바인딩이 없어 PyPI 휠 `pip install ompl`을 씁니다.)

## IsaacSim 시뮬레이션 실행 (반복 재생 데모)
`aerial_box_transport/` 디렉토리에서 두 단계로 실행합니다.

**1단계. 레퍼런스 궤적 생성** (am_dualarm, IsaacSim 불필요):
```
conda run -n am_dualarm python src/planner/ocp.py
```
OCP를 풀어 `results/ocp_reference.npz`(드론 베이스 + 팔 궤적, gRITE용 30차원 레퍼런스
포함)를 만듭니다. `results/`는 git에 포함되지 않으므로 시뮬레이션 전에 한 번은 실행해야
합니다.

**2단계. IsaacSim에서 재생** (am_isaac):
```
conda run -n am_isaac python -u src/sim/track_reference.py --max_time 11.5 --loop
```
GUI 창에서 드론이 책상의 박스를 집어 랙 맨 위 선반에 옮기는 장면이 반복 재생됩니다.
(팔 수평 출발 -> 박스 위로 비행하며 팔이 스스로 하강 -> base는 표면 위에 뜬 채 팔만
내려가 마찰로 파지 -> 운반 -> 선반에 내려놓기)

옵션:
- `--max_time <초>`: 재생 길이 (전체 궤적 약 10.8초; 기본 데모 11.5)
- `--loop`: 끝나면 씬을 리셋하고 처음부터 반복
- `--ref <경로>`: 다른 레퍼런스 npz 사용 (기본 `results/ocp_reference.npz`)

주의: 씬 에셋(로봇/책상/랙/박스 USD)은 `track_reference.py` 안에서 절대경로
`/home/jaewoo/Research/motion_planning_final/...`로 로드됩니다. 이 repo가 해당 경로에
클론되어 있어야 하며, 경로가 다르면 스크립트 상단의 `USD`, `_REPO` 변수를 수정하세요.
박스는 팀원의 USD를 수정하지 않고 우리 spawn 설정(`track_reference.py`)에서 z로 1.5배
키우고 질량을 1.0 kg으로 지정해, 박스 중심이 책상 위로 더 높이 떠 그리퍼가 책상 표면에
닿지 않게 했습니다.

## 단위테스트 / 헤드라인 실험 (IsaacSim 불필요, am_dualarm)
미끄럼 인지 힘 법칙이 고정력 베이스라인 대비 임계 가속도 이후에도 박스를 놓치지 않음을
보이는 가속도 스윕입니다.
```
conda run -n am_dualarm pytest -q
conda run -n am_dualarm python src/experiments/accel_sweep.py
```
그림과 요약은 `results/`에 저장됩니다.

## OCP 정식화 문서
비용 함수와 모든 제약조건(접촉 모델, 패드-표적 파지, 박스-base 정렬 비용, 미끄럼 인지
조건, 장애물 회피, base 클리어런스, 발견되는 수직 원기둥 경로 등)을 자세히 설명한 문서는
`docs/OCP_formulation.pdf`에 있습니다.

## 플래너 비교 연구 (OCP vs 샘플링 vs 하이브리드)
같은 박스 운반 문제를 세 가지 방법으로 풀고 비교합니다(논문용). 충돌 기하는 세 방법이
동일하게(OCP의 bounding-box keep-out) 봅니다.

- **OCP** (`ocp.py`, am_dualarm): 동역학·접촉·정렬 cost를 한 번에 푸는 전역 최적화.
  전체 시퀀스를 ~126초에 풀어 매끄럽고 외란 적은(<1°) 궤적 생성.
- **샘플링** (am_sampling): 14-DoF whole-body C-space에서 충돌 없는 기하 경로 탐색.
  - `sampling_compare.py --sweep`: OMPL 플래너 sweep. 운반 구간이 narrow passage라
    단일트리 RRT는 실패(30%)하고 양방향(RRTConnect/BKPIECE/LBKPIECE)은 100% 성공.
  - `grasp_constrained.py`: 물체를 든 채 자세를 바꿀 수 있도록 **파지 제약 manifold**
    위에서 운반 계획(CBiRRT). 그립을 고정하지 않고 팔이 재배치됨. box-under-base cost를
    Riemannian gradient descent로 후처리(method 5)하면 외란이 줄지만 ~16cm에서 바닥.
  - `sampling_to_reference.py`: 샘플 경로를 트래커 레퍼런스(`results/sampling_reference.npz`)로
    변환. `track_reference.py --ref` 로 IsaacSim에서 재생.
- **하이브리드** (`hybrid_seed_ocp.py`, build@am_sampling -> solve@am_dualarm): 샘플러가
  narrow passage 경로(homotopy)를 찾고, OCP가 그 box 경로로 warm-start돼 매끄러운 cost를
  refine. 핵심: OCP의 cylinder 제약을 끄고(`use_cylinders=False`) box 경로만 seed해야
  IPOPT가 수렴. 결과: 운반 중 box-base offset 0cm, tilt 0.43° (샘플링만 ~4-6° 대비).

실행 예:
```
conda run -n am_sampling python src/planner/sampling_compare.py --sweep --timeout 20 --trials 10
conda run -n am_sampling python src/planner/sampling_to_reference.py   # 샘플 경로 -> 레퍼런스
conda run -n am_sampling python src/planner/hybrid_seed_ocp.py --stage build
conda run -n am_dualarm  python src/planner/hybrid_seed_ocp.py --stage solve
conda run -n am_isaac    python src/sim/track_reference.py --ref results/hybrid_reference.npz --loop
```
요약: narrow passage 탐색은 샘플링이, 매끄러운 nonlinear cost(정렬 등) 최소화는 OCP가
유리하며, 하이브리드(샘플러 경로로 OCP warm-start)가 둘의 장점을 결합합니다. 단 하이브리드
handoff는 plug-and-play가 아니라 formulation을 맞춰야 합니다(샘플러 경로가 OCP의 cylinder를
대체, OCP가 자체 일관된 dynamic seed 생성).

## 창문 통과 + 키프레임 가이드 (메인 연구, ICRA/IROS 타겟)
박스를 들고 **부분 개방된 기울어진 차양(awning) 창문**을 통과하는, 전신 재구성이 필요한
narrow passage 문제입니다. 벽은 개구부를 제외하면 막혀 있고(비볼록), sash가 32.4° 기울어
방을 향해 돌출해 있어, 드론이 자세를 바꿔 박스를 대각선으로 비집어 넣어야 합니다.

**충돌 모델**(`sampling_compare.py`): 몸체를 coal 3.0.3(GJK/EPA)로 봅니다. 얇은 탄소
파이프 팔은 **캡슐**(축 선분 + 반지름), base는 납작한 **박스**, 박스는 **박스**로 두고,
창문은 개구부를 둘러싼 4개의 축정렬 벽 박스 + 기울어진 sash 박스로 봅니다. 구(sphere)
집합으로 감싸면 얇은 파이프를 과대평가해 통과 가능한 틈을 막아버립니다.

**창문 OCP**(`ocp.py`의 `solve_ocp(window=True, window_mode=...)`): 창문 keep-out을 미분
가능한 **soft penalty**로 넣습니다(hard 제약은 IPOPT multiplier가 발산). `window_mode`는
`soft`(점 샘플), `soft_box`(분석적 oriented-box support 함수)를 지원합니다. 윈도우 통과는
pick/place의 수직 cylinder 제약과 충돌하므로 **반드시 `use_cylinders=False`**로 풀어야 합니다.

**wedge keep-out**(`win_under`): sash의 얇은 판만 막지 않고, sash **윗 공간 전체**에 몸체가
들어가는 것을 금지합니다(`(p - s_c)·s_n <= -(h+r)`, sash x-구간으로 게이팅). 그러면 몸체가
sash 아래로 내려가 더 낮은 개구부를 통과하도록 자연스럽게 유도됩니다.

**키프레임 가이드**(메인 가설): 샘플러 대신 사람이 준 키프레임 한 개로 OCP를 안내합니다.
`build_kf_seed.py`(am_sampling)가 grasp->keyframe->over(rack)->place를 **제약 일관**하게
보간한 seed를 만들고(팔 좌우대칭 + 박스=FK(config)), `keyframe_window.py`(am_dualarm)가
키프레임 창 knot에 soft waypoint를 걸어 OCP를 풉니다. seed/가중치는 `KF_SEED`, `W_KF`,
`W_TRK_TH` 환경변수로 조절합니다.

**결과**(`verify_window.py`, am_sampling, coal 부호 거리로 세 레퍼런스를 동일 방식 검증):

| 지표 (149 knots) | soft_box(샘플러) | wedge_only | **keyframe** |
|---|---|---|---|
| base 최소 여유 | -3.8 cm | -0.9 cm | **+2.8 cm** |
| arm 최소 여유 | +0.2 cm | +3.3 cm | **+5.7 cm** |
| box 최소 여유 | +2.6 cm | +3.8 cm | **+4.1 cm** |
| 창 관통 knot | 2 / 149 | 2 / 149 | **0 / 149** |
| 최대 자세(rotvec norm) | 86.7° | 96.8° | **53.2°** |

핵심: 가이드 없는 해는 큰 자세가 대부분 **yaw**입니다. base를 옆으로 85~90° 돌려 몸의 긴
축을 모로 세워 비집고(yaw -81~-90°, pitch만 +16~+24°) 통과하지만 창틀을 긁습니다(관통 2회).
사람 키프레임 하나(pitch 60°, yaw 없음)는 **pitch +1~+47.5°**의 정면 대각선 통과로 유도해,
기울어진 차양에 몸을 기울여 들어갑니다. 결과는 **유일한 무충돌 궤적(0/149)**, 모든 부위
최고 여유, 그리고 훨씬 온건한 자세(53° vs 87~97°)입니다. 즉 키프레임은 기존 경로를 미세
조정한 게 아니라, 질적으로 다르고 더 나은 homotopy class를 선택합니다. 자세한 표와 해석은
`docs/window_keyframe_results.md`를 보세요.

실행 예:
```
# 1) 키프레임 보간 seed 생성 (am_sampling, FK 필요)
conda run -n am_sampling python src/planner/build_kf_seed.py
# 2) 키프레임 가이드 OCP (am_dualarm) -> results/window_reference_keyframe.npz
W_KF=40 W_TRK_TH=0 conda run -n am_dualarm python src/planner/keyframe_window.py
# 3) 세 레퍼런스 coal 정량 비교 (am_sampling)
conda run -n am_sampling python src/planner/verify_window.py
# 4a) 계획 경로 기구학적 재생 (am_isaac, 창문 keep-out 볼륨 표시)
conda run -n am_sampling python src/planner/export_play.py /tmp/window_kf_path.npy \
    /tmp/window_kf_play.npz --box results/window_reference_keyframe.npz
conda run -n am_isaac python src/sim/play_path.py --play /tmp/window_kf_play.npz --proxy --loop
# 4b) gRITE 동역학 추종 (am_isaac, 동적 박스 + 실제 창 충돌체)
conda run -n am_isaac python src/sim/track_reference.py \
    --ref results/window_reference_keyframe.npz --loop --max_time 15
```

gRITE 동역학 추종 결과: 창 통과 순간(최대 pitch 53.5°) base 위치 오차 약 1.9 cm, arm
오차 0.04~0.07 rad로 그립을 유지한 채 통과합니다. 즉 키프레임 경로는 기하적 무충돌일 뿐
아니라 실제 컨트롤러로 동역학적으로도 추종 가능합니다.
