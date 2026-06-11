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
수행하는 전신 궤적을 한 번에 생성합니다.

- **접근**: 팔을 수평으로 편 채 출발해 박스 바로 위로 비행한 뒤 수직으로 하강
- **파지**: base는 책상 표면 위로 일정 거리 떠 있고(지상효과/충돌 회피), 팔만 대각선
  아래로 내려가 박스에 닿음. 두 그리퍼 패드가 **평행하게** 박스 양면을 누르고, 가속도에
  맞춘 스퀴즈 힘으로 **마찰 파지**(미끄럼 방지)
- **운반**: 책상과 랙을 바운딩 박스 제약으로 회피하며 위로 올라가 랙 맨 위 선반으로 이동
- **놓기**: 선반 위에서 그리퍼를 열어 박스를 내려놓음

팔이 아래로 내려가는 동작은 명시적으로 지정한 것이 아니라, "base는 표면에서 일정 거리
이상 떨어져 있어야 한다"는 제약과 "박스를 잡아야 한다"는 조건으로부터 최적화가 스스로
찾아낸 결과입니다.

## 디렉토리 구조
- `config/robot.yaml`  접촉 강성 k, 평활화 eps, 마찰계수 mu, 스퀴즈 마진
- `config/task.yaml`   박스 질량/크기, pick/place, 장애물(책상/랙) 바운딩 박스, 팔 설정
  (arm_grasp/arm_pregrasp, dof1 포함), base_clearance, 단계별 시간, 가속도 스윕
- `config/usd_model.json`, `config/isaac_model.json`  USD/PhysX에서 추출한 운동학/관성
- `src/model/`     전신 모델(`whole_body.py`, USD->Pinocchio), 평활 접촉(`contact.py`),
  박스/마찰콘(`box.py`), USD/그래스프 분석 도구
- `src/planner/`   다단계 OCP(`ocp.py`), 미끄럼 인지 힘 법칙(`slip_aware.py`),
  시간 이산화(`transcription.py`), OCP 문서 생성(`make_ocp_pdf.py`)
- `src/sim/`       gRITE SE(3) 컨트롤러(`grite_controller.py`), IsaacSim 재생/검증
  (`track_reference.py`)
- `src/baselines/`, `src/experiments/`  고정력 베이스라인, 가속도 스윕/헤드라인 플롯
- `tests/`         플래너 단위테스트(pytest)
- `docs/`          `OCP_formulation.pdf` (비용 함수 + 제약조건 상세 설명, 한국어)
- `results/`       생성 산출물(궤적 npz, 그림). git에는 포함되지 않음(재생성 가능)

## 환경 (conda 2개)
- **`am_dualarm`**: 플래너용. Pinocchio(+pinocchio.casadi), CasADi, IPOPT, numpy,
  scipy, matplotlib, pyyaml, pytest. OCP를 풀어 레퍼런스 궤적을 생성
- **`am_isaac`**: 시뮬레이션용. IsaacSim 5.1 / IsaacLab. 생성된 궤적을 GUI에서 재생

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

## 단위테스트 / 헤드라인 실험 (IsaacSim 불필요, am_dualarm)
미끄럼 인지 힘 법칙이 고정력 베이스라인 대비 임계 가속도 이후에도 박스를 놓치지 않음을
보이는 가속도 스윕입니다.
```
conda run -n am_dualarm pytest -q
conda run -n am_dualarm python src/experiments/accel_sweep.py
```
그림과 요약은 `results/`에 저장됩니다.

## OCP 정식화 문서
비용 함수와 모든 제약조건(접촉 모델, 미끄럼 인지 조건, 장애물 회피, base 클리어런스 등)을
자세히 설명한 문서는 `docs/OCP_formulation.pdf`에 있습니다.
