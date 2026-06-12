"""OCP의 비용함수와 제약조건을 자세히 설명하는 PDF 생성 (한국어).

이 머신에는 LaTeX/pandoc이 없어서 matplotlib mathtext(LaTeX 부분집합)로 수식을
조판하고, 한글 본문은 NanumGothic 폰트로 렌더한다. 내용은 src/planner/ocp.py와 일치.

실행: conda run -n am_dualarm python src/planner/make_ocp_pdf.py
출력: docs/OCP_formulation.pdf
"""
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager as fm, rcParams  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

for _cand in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
              "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"):
    if os.path.exists(_cand):
        fm.fontManager.addfont(_cand)
rcParams["font.family"] = "NanumBarunGothic"
rcParams["mathtext.fontset"] = "dejavusans"   # 수식은 DejaVu math (한글 폰트와 무관)
rcParams["axes.unicode_minus"] = False

# (type, text). types: h1, h1b, h2, body, eq, bul, gap
BLOCKS = [
    ("h1", "접촉 인지 전신 궤적 최적화"),
    ("h1b", "듀얼암 공중 매니퓰레이터 박스 운반  :  OCP 정식화"),
    ("body", "하나의 비선형 최적제어문제(OCP, CasADi Opti + IPOPT)를 고정 horizon에서 푼다. "
             "결과로 전신 reference 궤적(완전구동 6-DoF 베이스 + 8개 팔관절)이 나오며, 이는 "
             "책상 위 박스로 날아가 마찰로 잡고, 운반해, 랙 맨 위 선반에 놓는 동작이다."),
    ("body", "핵심 설계는 '경로를 미리 주지 않는다'이다. 베이스의 비행 경로, 팔이 내려가는 동작, "
             "박스의 up-over-down 운반 경로는 어느 것도 reference로 추종하지 않는다. 대신 (a) 패드가 "
             "있어야 할 위치(박스 면 중심)를 핀하는 제약, (b) 박스를 좁은 수직 원기둥에 가두는 제약, "
             "(c) 장애물 keep-out 제약, (d) 박스를 베이스 바로 아래로 끌어당기는 정렬 비용이 함께 "
             "작용해, 최적화가 경로 전체를 스스로 발견한다(가이드 없음, IPOPT 초기 seed만 둠). "
             "Transcription은 가속도 레벨이고 semi-implicit Euler 적분을 쓰며, 베이스 렌치/로터 "
             "allocation은 하류의 gRITE 제어기에 위임한다(결정 D5)."),
    ("gap", 0.4),

    ("h2", "1.  설정, 좌표계, horizon, phase FSM"),
    ("body", "모든 양은 HOME 좌표계 기준이고, 원점은 드론 스폰 지점(월드 (0,0,1.5) m)이다. "
             "베이스 자세는 회전벡터 theta in R^3(지수좌표)로 매개화하고 유클리드 공간에서 "
             "적분한 뒤, forward kinematics 입력에만 단위 쿼터니언으로 변환한다(단위노름 manifold "
             "등식을 피해 IPOPT 발산을 막음)."),
    ("body", "task는 박스의 두 점 pick과 place(각각 집을 때/놓을 때의 박스 중심)로만 주어진다. 어디서 "
             "어떻게 잡을지(베이스 위치, 팔 형상)나 어떤 경로로 운반할지는 주지 않으며, 5절의 제약과 "
             "비용에서 발견된다. 박스는 z로 1.5배 높은(taller) 크기라 중심이 책상 위 약 8 cm에 있어 "
             "그리퍼가 책상 표면에서 여유 있게(EE tracking 오차에도 책상에 안 닿게) 잡는다."),
    ("body", "Phase FSM: 4개 phase(approach 3.0, grasp 1.6, transport 5.0, release 1.2 s)는 "
             "고정 시간 스케줄로 정해지는 단순한 시간 인덱스 FSM이다. phase_of[k]가 각 제어구간 k의 "
             "phase를 정하고, 비용과 제약이 그에 따라 분기한다(이벤트가 아니라 시간으로 전환). "
             "dynamics와 접촉 모델은 모든 phase에서 동일하고 매끄럽다(경계에서 스위칭 없음). "
             "dt = 0.1 s -> N = 108 구간, N+1 = 109 knot."),
    ("gap", 0.4),

    ("h2", "2.  결정 변수 (각 knot k = 0..N)"),
    ("eq", r"$q_k=[\,p_k,\ \theta_k,\ \alpha_k\,]\in\mathrm{R}^{14}$"
           r"   베이스 위치 $p$, 자세 회전벡터 $\theta$, 팔관절 $\alpha$"),
    ("eq", r"$v_k=[\,\dot p_k,\ \omega_k,\ \dot\alpha_k\,]\in\mathrm{R}^{14}$"
           r"   베이스 선속도 / 각속도 / 팔 관절속도"),
    ("eq", r"$u_k\in\mathrm{R}^{14}\ (k=0..N\!-\!1)$  가속도      "
           r"$b_k\in\mathrm{R}^{3}$  박스 위치"),
    ("body", "FK는 전체 형상 q_full(q_k) = [ p_k, exp(theta_k), alpha_k ]를 쓴다(exp()는 "
             "회전벡터를 쿼터니언으로). 팔은 8관절(팔당 4개): 관절1(dof1)은 연속(무한) 관절로 "
             "두 팔에 반대부호를 주어 팔 전체를 아래로 기울이고(reach-down, 5c), 관절2-4는 [0,pi]. "
             "박스 위치 b_k도 결정변수이며 phase별 제약으로 묶인다(5절). Pinocchio 모델은 USD에서 "
             "만들고 Isaac과 0.00 mm 일치 검증함."),
    ("gap", 0.4),

    ("h2", "3.  접촉 모델 (결정변수가 아니라 유도량)"),
    ("body", "두 패드가 박스를 월드 x축으로 누른다(팔이 아래로 뻗어도 패드 면은 평행 그리퍼 제약 "
             "5c로 월드 x에 수직 유지되어 스퀴즈 축은 여전히 월드 x). EE 위치 (p^L, p^R)는 FK로 "
             "구한다. 가상 침투량 phi(양수=분리, 음수=침투)로 매끄러운 수직력 lambda를 정의한다 "
             "(결정 D2/D3: 침투에서 유도, 상보성(complementarity) 없음, 접촉력 결정변수 없음):"),
    ("eq", r"$\varphi_L=(b_x-p^{L}_{x})-h,\qquad \varphi_R=(p^{R}_{x}-b_x)-h$"),
    ("eq", r"$\lambda(\varphi)=\frac{1}{2}\,k_c\left(\sqrt{\varphi^{2}+\epsilon^{2}}-\varphi\right)$"),
    ("body", "분리 시(phi>0) lambda->0, 침투 시(phi<0) lambda->-k_c*phi이며 phi=0에서 도함수가 "
             "연속이다. 파라미터: k_c = 2000 N/m, epsilon = 1e-4 m, 반폭 h = 0.0525 m (x로 10.5 cm)."),
    ("gap", 0.4),

    ("h2", "4.  비용함수  J = sum_k ( phase별 항 )  +  공통 항"),
    ("body", "잡는 위치는 패드를 박스 면에 핀하는 제약(5절)이, 운반/비행 경로는 cylinder/keep-out "
             "제약(5b,5d)이 정한다. 따라서 비용은 어떤 베이스/팔/박스 pose도 추종하지 않고(과거의 "
             "base_grasp / arm_grasp / box_guess 추종은 모두 제거), 스퀴즈 세기, 좌우 대칭, 그리퍼 "
             "열림, 박스-베이스 정렬, 자세, 노력(effort)만 담당한다. lambda_L, lambda_R은 3절 패드 수직력."),
    ("eq", r"$J_{f}=w_{f}[(\lambda_L-F^{*})^{2}+(\lambda_R-F^{*})^{2}]$"
           r"   스퀴즈를 $F^{*}$로 (grasp/transport 22 N, release 2 N)"),
    ("eq", r"$J_{sym}=w_{sym}\,(\lambda_L-\lambda_R)^{2}$    좌우 스퀴즈 대칭 (grasp/transport)"),
    ("eq", r"$J_{open}=w_{h}\,\| \alpha^{2:4}_k-\alpha^{pre}_{2:4}\|^{2}$"
           r"   그리퍼 열림 유지 (approach; dof1 자유)"),
    ("eq", r"$J_{align}=w_{al}[(b_{x,k}-p_{x,k})^{2}+(b_{y,k}-p_{y,k})^{2}]$"
           r"   박스를 베이스 바로 아래로 (모든 phase)"),
    ("eq", r"$J_{att}=w_{att}\,\|\theta_k\|^{2}$    베이스 수평 유지 (모든 phase)"),
    ("eq", r"$J_{reg}=w_{a}\,\| u_k\|^{2}+w_{v}\,\| v_k\|^{2}$"
           r"    가속도 / 속도 최소화 (모든 phase)"),
    ("body", "J_align(박스-베이스 정렬): 물체가 베이스 중심을 지나는 수직선(global z) 위에 있도록 "
             "수평 offset을 벌점한다. omnidirectional 드론에서 물체가 베이스 중심에서 멀수록 외란 "
             "토크와 CoM 이동이 커지므로 이를 최소화한다. 이는 waypoint 가이드가 아니라 단순 벌점이며, "
             "approach에서는 이 항이 베이스를 박스 위로 끌어 (제거한 비행 가이드를 대체해) 비행 경로를 "
             "발견하게 한다(5d). 가중치: w_f = 1, w_sym = 2, w_h = 2, w_al = 50, w_att = 30, "
             "w_a = 1e-3, w_v = 1e-2. 종단 w_att||theta_N||^2 추가."),
    ("gap", 0.4),

    ("h2", "5.  제약조건"),
    ("bul", "초기조건:  q_0 = [ home, theta = 0, alpha_start ],  v_0 = 0.  alpha_start는 팔이 "
            "안 내려간 수평 자세(dof1 = 0, 그리퍼는 열림)."),
    ("eq", r"동역학 $(k=0..N\!-\!1)$:   $v_{k+1}=v_k+u_k\,\Delta t,\quad "
           r"q_{k+1}=q_k+v_{k+1}\,\Delta t$"),
    ("bul", "phase별 박스 위치 (no-slip carried-box 모델, 결정 D6):"),
    ("eq", r"approach, grasp:  $b_k=\mathrm{pick}$       (박스가 책상 지지대에 놓임)"),
    ("eq", r"transport:  $b_{x,k}=0.5\,(p^{L}_{x,k}+p^{R}_{x,k})$   (박스 x를 EE 중점에)"),
    ("eq", r"release:  $b_k=\mathrm{place}$;     종단:  $b_N=\mathrm{place}$"),
    ("bul", "패드-표적 파지 (grasp/transport/release): task가 주는 것은 각 패드의 표적 위치, 곧 "
            "박스 좌/우 면의 중심뿐이다. 각 패드의 y,z를 박스 면 중심에 핀한다(x는 스퀴즈가 누름):"),
    ("eq", r"$p^{L}_{y}=b_y,\ \ p^{L}_{z}=b_z,\qquad p^{R}_{y}=b_y,\ \ p^{R}_{z}=b_z$"),
    ("body", "따라서 OCP는 이 두 패드 표적에 도달하는 베이스+팔을 스스로 발견한다(precompute한 "
             "base_grasp / arm_grasp pose를 추종하지 않음). transport에서는 박스 y,z가 패드를 따라온다. "
             "EE의 위치만 알면 whole-body planning이 베이스 궤적을 자동으로 정한다는 것이 핵심."),
    ("bul", "수직 강하 straddle (approach 마지막 ~45%): 닫기 전 넓게 벌린 패드의 중점을 박스 바로 "
            "위에 둬, 패드가 박스를 정 가운데 두고 수직으로 내려온다(앞으로 쓸어 박스를 밀지 않음):"),
    ("eq", r"$0.5\,(p^{L}_{x}+p^{R}_{x})=b_x,\qquad 0.5\,(p^{L}_{y}+p^{R}_{y})=b_y$"),
    ("bul", "좌우 대칭 (모든 knot, k >= 1): 두 팔은 거울상이어야 한다(없으면 왼팔이 박스 앞, "
            "오른팔이 박스 뒤로 가는 비대칭 해가 핀을 만족하며 생긴다):"),
    ("eq", r"$\alpha^{1}_L=-\alpha^{1}_R,\quad \alpha^{2}_L=\alpha^{2}_R,"
           r"\quad \alpha^{3}_L=\alpha^{3}_R,\quad \alpha^{4}_L=\alpha^{4}_R$"),
    ("bul", "평행 그리퍼 (모든 knot, k >= 1): 패드 면이 월드 x에 수직이도록, 면 각도(팔당 "
            "d2-d3-d4)를 0으로 강제한다. pregrasp/grasp 모두에서 0이라 닫힘이 순수 스퀴즈다:"),
    ("eq", r"$\alpha^{2}_k-\alpha^{3}_k-\alpha^{4}_k=0$    (왼팔; 오른팔은 대칭으로 따라옴)"),
    ("bul", "slip-aware 스퀴즈 (transport 구간) : 헤드라인 법칙, HARD 부등식:"),
    ("eq", r"$\lambda_L\geq f_{set,k},\qquad \lambda_R\geq f_{set,k}$"),
    ("eq", r"$f_{set,k}=\max\!\left(\frac{m_o\,(g+a_{z,k})}{2\mu}+\mathrm{margin},\ \mathrm{floor}\right)$"),
    ("body", "a_{z,k}는 계획된 박스 수직 가속도(seed 박스 z의 2차 차분). m_o = 1.0 kg, "
             "g = 9.81, mu = 0.7, margin = 0.5 N, floor = 2 N. 따라서 마찰 원뿔이 중력 + 운반 "
             "가속도에 대해 박스를 항상 잡을 수 있다."),
    ("bul", "경계 (모든 knot):  dof2,3,4 in [0, pi], dof1 자유;   -3 <= v_k <= 3 ;   -25 <= u_k <= 25."),
    ("gap", 0.45),

    ("h2", "5b.  장애물 keep-out (HARD; 박스 + 드론 베이스; transport)"),
    ("body", "점 p가 footprint [x_lo,x_hi] x [y_lo,y_hi] 위에 있는지의 매끄러운 indicator:"),
    ("eq", r"$I(p)=\sigma(p_x;x_{lo},x_{hi})\,\sigma(p_y;y_{lo},y_{hi}),\quad "
           r"\sigma(v;a,b)=\frac{1}{2}[\tanh\frac{v-a}{s}-\tanh\frac{v-b}{s}]$"),
    ("body", "각각 >= 0 인 잔차 3개 (M은 점이 그 footprint 위에 없을 때 잔차를 비활성화하는 "
             "smooth big-M):"),
    ("eq", r"$p_z+M\,(1-I_{desk}(p))\ \geq\ z_{desk}$           (책상 위로 비켜감)"),
    ("eq", r"$p_z+M\,(1-I_{rack}(p))\ \geq\ z_{shelf}$          (랙 위에서 top shelf 위)"),
    ("eq", r"$p_z+M\,(1-I_{rack}(p)(1-I_{slot}(p)))\ \geq\ z_{clear}$   (랙 프레임 위, 중앙 slot 제외)"),
    ("body", "footprint는 body margin bm = 0.35 m로 팽창시켜 박스 중심뿐 아니라 드론+박스 전체가 "
             "비켜가게 한다. M = 1.5, indicator 폭 s = 0.04. z_desk, z_shelf = (장애물 top + 박스 "
             "반높이); z_clear = (랙 프레임 2.0 + 박스 반높이 + 0.1). 중앙 landing slot은 '프레임 위' "
             "잔차만 랙 중앙에서 완화하므로, 박스는 랙 접근에서 높이 올라갔다가 slot으로 선반에 내려온다. "
             "transport에서 박스와 드론 베이스 양쪽에 적용된다."),
    ("gap", 0.4),

    ("h2", "5c.  베이스 클리어런스 · EE-표면 위 · 자연스러운 팔 내림"),
    ("body", "베이스가 책상/랙 표면에 너무 가까우면 지상효과가 크고 작은 오차로도 충돌한다. 그래서 "
             "드론 베이스(팔 제외)는 표면 위로 최소 c_lr 이상 떠 있어야 한다(모든 knot). 5b와 같은 "
             "indicator를 쓰되 박스 반높이 없이 표면 자체(z^surf = 장애물 top)까지 잰다:"),
    ("eq", r"$p_z+M\,(1-I_{desk}(p))\ \geq\ z^{surf}_{desk}+c_{lr}$    (책상 위 $c_{lr}$ 이상)"),
    ("eq", r"$p_z+M\,(1-I_{rack}(p))\ \geq\ z^{surf}_{rack}+c_{lr}$   (랙 선반 위 $c_{lr}$ 이상)"),
    ("body", "또한 각 패드를 반지름 r_ee의 작은 구로 보고, 수직 강하 중 표면 아래로 파고들지 않게 "
             "표면 위에 둔다(모든 knot, 양 패드). r_ee = 0.06 m는 박스가 표면 위로 나온 높이보다 작아 "
             "패드는 여전히 박스 면(grip 높이)에 닿는다."),
    ("eq", r"$p^{pad}_z+M\,(1-I_{desk})\ \geq\ z^{surf}_{desk}+r_{ee}$   (랙도 동일)"),
    ("body", "자연스러운 팔 내림(핵심): dof1에 내림 reference를 주지 않는다. c_lr = 0.30 m로 베이스는 "
             "위에 떠 있어야 하고, 패드는 표면의 박스 면에 핀돼 있으므로, 팔이 아래로 뻗는 것이 유일한 "
             "해가 되어 최적화가 dof1을 스스로 내린다. 관절1을 두 팔에 반대부호로 주면 두 팔이 대칭으로 "
             "내려가며 패드 x(파지 폭)와 월드 x 면 법선은 보존하고 수직 위치만 낮춘다. 평행 그리퍼 "
             "식(d2-d3-d4=0)을 dof1이 보존하므로, 팔을 내려도 면은 계속 월드 x에 수직(평행)이다."),
    ("gap", 0.4),

    ("h2", "5d.  발견되는 경로: 수직 원기둥 (HARD; 가이드 없음)"),
    ("body", "운반 경로에 가이드를 주지 않는다(과거의 up-and-over guide box_guess 제거). 대신 박스 "
             "중심만 pick 위와 place 위의 좁은 수직 원기둥 안에 가둬, 그 높이 구간에서는 수직으로 "
             "오르내리게 한다. 원기둥 위/밖에서는 자유라, keep-out(5b)과 합쳐 최적화가 up-over-down을 "
             "스스로 발견한다(팔/드론은 어디에 있어도 됨, 박스 중심만 제약). 모든 knot에서:"),
    ("eq", r"$r_c^{2}+M_c\,(1-a)-[(p_x-c_x)^2+(p_y-c_y)^2]\ \geq\ 0,\quad a=g_x\,(1-s_z)$"),
    ("eq", r"$g_x=\frac{1}{2}\!\left(1+\tanh\frac{\mathrm{side}\,(p_x-x_{mid})}{0.30}\right)$"
           r"   (미드포인트의 이 쪽이면 ~1)"),
    ("eq", r"$s_z=\frac{1}{2}\!\left(1+\tanh\frac{p_z-z_{top}}{0.04}\right)$"
           r"   ($z_{top}$ 아래면 ~0, 위면 ~1)"),
    ("body", "pick 위(side = +1, 중심 = pick)와 place 위(side = -1, 중심 = place) 두 원기둥. 활성도 "
             "a = g_x(1 - s_z)는 '미드포인트의 그 쪽 AND z_top 아래'에서만 1이라 박스를 d <= r_c "
             "기둥에 가둔다. side로 게이트(반지름이 아니라)해서 게이트가 좁은 기둥 전체에서 ~1로 "
             "유지되므로 박스가 옆으로 새지 못하고, 풀리려면 z_top까지 수직으로 올라야 한다. "
             "r_c = 0.10 m, M_c = 25, cyl_h = 0.50 m (z_top = 표면 + cyl_h)."),
    ("body", "발견되는 approach 비행: approach에도 베이스 경로 reference가 없다. 정렬 비용 J_align(4절, "
             "모든 phase)이 베이스를 박스 위로 끌고, 패드-표적 핀과 straddle 제약이 아래로 끌어, "
             "over-then-down 강하가 비용/제약의 결과로 나온다. 결국 비행/팔내림/운반 어느 경로도 "
             "미리 주지 않으며 모두 발견된다."),
    ("gap", 0.4),

    ("h2", "6.  초기 seed, solver"),
    ("body", "가이드와 seed의 구분: base_ref(over-then-down), box_guess(up-and-over), 팔 reference는 "
             "더 이상 비용으로 추종하지 않고(가이드 아님) 오직 IPOPT 초기 guess(seed)로만 쓴다. seed로 "
             "쓰는 베이스 위치는 base_grasp = pick - r_off이며(r_off는 grasp 형상 FK로 한 번 계산, "
             "수평 y성분은 0으로) 정렬 비용이 같은 곳으로 끌어 일관적이다. 영속도 guess는 metre 스케일 "
             "비행에서 IPOPT restoration에 실패하므로, 위치/속도/가속도가 일관된 전체 운동학 seed를 준다."),
    ("body", "팔 seed: arm_start(수평, dof1 = 0; IC) -> arm_pre(reach-down 열림, ~30 cm gap) during "
             "traverse -> arm_grasp(reach-down 닫힘, ~10.5 cm gap). 넓게 연 ~30 cm gap은 작은 박스를 "
             "패드가 nick하지 않게 여유를 준다. Solver: IPOPT via CasADi Opti, max_iter 5000, "
             "tol 1e-4, acceptable_tol 1e-3, adaptive mu. 출력: 박스/베이스/팔 궤적과 30차원 gRITE "
             "reference [pos, vel, acc, jerk, R, omega, omega_dot, omega_ddot]."),
]

PAGE_W, PAGE_H = 8.27, 11.69      # A4 inches
LEFT, RIGHT, TOP, BOT = 0.085, 0.93, 0.945, 0.06
LH = 0.0155


def dwidth(s):                    # display width: 한글/CJK 2, ASCII 1
    return sum(2 if ord(ch) >= 0x1100 else 1 for ch in s)


def kwrap(text, budget=96):
    out = []
    for para in text.split("\n"):
        cur = ""
        for w in para.split(" "):
            trial = (cur + " " + w) if cur else w
            if cur and dwidth(trial) > budget:
                out.append(cur); cur = w
            else:
                cur = trial
        out.append(cur)
    return out


def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "docs"))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "OCP_formulation.pdf")
    with PdfPages(path) as pdf:
        fig = plt.figure(figsize=(PAGE_W, PAGE_H)); y = [TOP]

        def newpage():
            nonlocal fig
            pdf.savefig(fig); plt.close(fig)
            fig = plt.figure(figsize=(PAGE_W, PAGE_H)); y[0] = TOP

        def put(s, x, fontsize, weight="normal", color="black", ha="left"):
            fig.text(x, y[0], s, fontsize=fontsize, weight=weight, color=color, ha=ha, va="top")

        for kind, text in BLOCKS:
            if kind == "gap":
                y[0] -= text * LH
                continue
            if kind == "h1":
                if y[0] < 0.5:
                    newpage()
                put(text, 0.5, 17, weight="bold", ha="center"); y[0] -= 0.036
            elif kind == "h1b":
                put(text, 0.5, 12, weight="bold", ha="center", color="#333333"); y[0] -= 0.042
            elif kind == "h2":
                if y[0] < BOT + 0.09:
                    newpage()
                y[0] -= 0.008
                put(text, LEFT, 13.5, weight="bold", color="#11337a"); y[0] -= 0.028
            elif kind == "eq":
                eqlh = 0.031 if ("\\frac" in text or "\\sqrt" in text) else 0.021
                if y[0] < BOT + eqlh:
                    newpage()
                put("    " + text, LEFT, 11.5, color="#7a1111"); y[0] -= eqlh + 0.004
            else:  # body, bul
                lines = kwrap(text, 92 if kind == "bul" else 96)
                indent = LEFT + (0.018 if kind == "bul" else 0.0)
                for i, ln in enumerate(lines):
                    if y[0] < BOT:
                        newpage()
                    put((("•  " if i == 0 else "   ") if kind == "bul" else "") + ln, indent, 9.7)
                    y[0] -= LH
                y[0] -= 0.004
        pdf.savefig(fig); plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
