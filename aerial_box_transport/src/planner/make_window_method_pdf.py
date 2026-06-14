"""현재 시뮬레이션(좁은 경사창 통과 전신 운반)의 방법론을 자세히 설명하는 PDF (한국어).

LaTeX/pandoc이 없어 matplotlib mathtext(LaTeX 부분집합)로 수식을 조판하고, 한글 본문은
NanumBarunGothic 폰트로 렌더한다. 내용은 src/planner/{ocp.py, sampling_compare.py,
grasp_constrained.py, hybrid_window.py} 와 src/sim/{grite_controller.py, track_reference.py}
의 실제 구현과 일치한다.

실행: conda run -n am_isaac python src/planner/make_window_method_pdf.py
출력: docs/window_method_ko.pdf
"""
import os

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
rcParams["mathtext.fontset"] = "dejavusans"
rcParams["axes.unicode_minus"] = False

# (type, text). types: h1, h1b, h2, h3, body, eq, bul, gap
BLOCKS = [
    ("h1", "좁은 경사창 통과 전신 운반 : 방법론"),
    ("h1b", "듀얼암 공중 매니퓰레이터  ::  샘플링 + 전신 궤적 최적화 + gRITE 추종  (현재 시뮬레이션)"),

    ("body", "현재 IsaacSim에서 돌고 있는 시뮬레이션은, 완전구동(omnidirectional) 공중 베이스에 두 "
             "팔이 달린 듀얼암 매니퓰레이터가 책상 위의 박스를 잡아, 위로 살짝 열린 경사진 어닝(awning) "
             "창문 틈으로 전신을 재구성하며 통과시켜, 벽 반대편 랙 선반에 내려놓는 전 과정을 다룬다. "
             "방법은 세 단계 파이프라인이다: (1) 샘플링 기반 경로 탐색이 충돌 없는 위상(homotopy)을 찾고, "
             "(2) 전신 궤적 최적화(OCP)가 이를 동역학적으로 실현 가능하고 미끄럼(slip)을 고려한 reference "
             "궤적으로 정제하며, (3) gRITE 기하-강건 제어기가 그 reference를 실제 물리 시뮬레이션에서 "
             "추종한다."),
    ("body", "핵심 설계 원칙은 '경로를 가이드로 주입하지 않는다'이다. 베이스가 어디로 날지, 박스를 어떻게 "
             "기울여 통과시킬지를 미리 정한 reference로 추종하는 대신, 비용(cost)과 제약(constraint)만으로 "
             "행동을 자연스럽게 유도한다. 특히 창문 통과에 필요한 베이스 틸트와 yaw 회전은 창문 keep-out "
             "제약을 만족시키는 결과로 스스로 나타난다."),
    ("gap", 0.3),

    ("h2", "1.  문제 설정과 좌표계"),
    ("body", "모든 양은 HOME 좌표계 기준이다 (원점 = 드론 스폰 지점, 월드 (0,0,1.5) m; 즉 "
             "home = world - (0,0,1.5)). 베이스 자세는 회전벡터 theta in R^3(지수좌표)로 매개화해 "
             "유클리드 공간에서 적분하고, forward kinematics 입력에만 단위 쿼터니언 exp(theta)로 변환한다 "
             "(단위노름 manifold 등식을 피해 IPOPT 발산을 막음)."),
    ("body", "task는 박스의 두 점 pick = (2.0, 0, -0.636)과 place = (-4.0, 0, 0.489)로만 주어진다 "
             "(각각 집을 때 / 놓을 때의 박스 중심, home 좌표). 랙을 벽에서 충분히 멀리(x = -4) 두어, "
             "드론이 창문을 완전히 빠져나간 뒤 선반에 놓도록 했다."),
    ("body", "창문(어닝) 형상: 벽은 x in [-1.10, -1.00] 두께의 평면이고 그 안에 사각 개구부가 뚫려 있다 "
             "(y in [-0.89, 0.89], z(월드) in [2.06, 3.07], home z in [0.56, 1.57]). 위쪽 경첩에서 "
             "바깥-아래로 32.4도 기울어진 sash(차양판; slant 1.168, width 1.78, thickness 0.041 m)가 "
             "개구부를 부분적으로 막아, 박스를 수평으로 그냥 밀어넣을 수 없고 베이스를 기울여 비스듬히 "
             "통과해야 한다."),
    ("body", "Phase FSM: approach -> grasp -> transport -> release 4개 phase를 고정 시간 스케줄로 정한다 "
             "(transport 시작 4.6 s, release 시작 13.6 s, 총 ~14.8 s). dt ~ 0.1 s, N ~ 148 구간. "
             "동역학과 접촉 모델은 모든 phase에서 동일하고 매끄럽다."),
    ("gap", 0.3),

    ("h2", "2.  충돌 모델 (coal GJK)"),
    ("body", "충돌은 coal 3.0.3(Pinocchio의 GJK/EPA 백엔드)로 정확히 계산한다. 얇은 탄소파이프 팔을 "
             "구(sphere)로 감싸면 과보수적이어서 좁은 틈을 거짓으로 막으므로, 몸체를 다음과 같이 모델한다:"),
    ("bul", "드론 베이스 = 납작한 박스 (0.5 x 0.5 x 0.16 m)."),
    ("bul", "각 팔 링크 = 캡슐 (선분 + 실제 파이프 반지름; 원기둥에 정확하고 tight한 Minkowski 합)."),
    ("bul", "운반 박스 = 박스 (0.106 x 0.105 x 0.158 m), 평행 그리퍼 강체 파지라 베이스 자세를 그대로 따른다."),
    ("body", "창문 장애물: 벽은 구멍이 뚫려 볼록(convex)이 아니므로 개구부 둘레를 4개의 축정렬 경계 "
             "박스(아래/위/왼/오른)로 타일링하고, 기울어진 sash 박스 1개를 더한다(총 5개). 모든 좌표는 "
             "home 프레임(-1.5 z shift)으로 일관되게 둔다. coal geom은 한 번 만들고 매 질의마다 자세만 "
             "갱신(mutate)하며, DistanceResult는 질의 전 clear()한다(상태가 남으면 직전 침투값이 오염)."),
    ("gap", 0.3),

    ("h2", "3.  1단계 - 샘플링 기반 경로 탐색 (제약 manifold CBiRRT)"),
    ("body", "좁은 통과는 RRT-Connect 계열 양방향 샘플러로 푼다. 단, 박스를 든 채 통과해야 하므로 자유 "
             "14-DoF가 아니라 '파지 제약 manifold' 위에서 계획한다."),
    ("body", "핵심 운동학 통찰: 평행 그리퍼 강체 파지에서 박스의 베이스-상대 자세는 그리퍼가 고정하므로, "
             "박스는 베이스가 회전할 때만 월드에서 회전한다. 따라서 파지 제약을 베이스 프레임(팔만의 FK, "
             "베이스 = 단위자세)으로 표현하면 베이스 자세에 불변이 된다(무작위 SO(3) 틸트 300개에서 패드 "
             "간격 / 제약이 정확히 불변임을 검증). 그 위에서 베이스 자세를 자유롭게(rotvec +-pi) 둔다."),
    ("body", "OMPL ProjectedStateSpace의 제약 State가 Python에서 설정 불가하여 투영을 직접 구현한다: "
             "q <- q - pinv(J) f (수치 J는 fk14 차분), 각 확장 스텝마다 manifold로 재투영하는 "
             "RRT-Connect. 충돌은 2절 coal 모델로 검사. 결과: 어닝 창문 통과를 ~2 초(약 470 노드)에 해결 "
             "(베이스 x +2 -> -2로 벽을 통과, 파지 잔차 ~1e-4). 이 경로가 OCP의 warm-start seed가 된다."),
    ("gap", 0.3),

    ("h2", "4.  2단계 - 전신 궤적 최적화 (OCP)"),
    ("body", "하나의 비선형 OCP(CasADi Opti + IPOPT)를 고정 horizon에서 푼다. 샘플러 경로를 초기 "
             "guess(및 베이스 / 팔의 약한 추종)로 쓰되, 동역학 / 접촉 / 창문 keep-out / 자세 비용을 함께 "
             "부과해 동역학적으로 실현 가능한 reference로 정제한다. 샘플러는 위상을 찾고, OCP는 그것을 "
             "재발견하지 않고 정제한다."),

    ("h3", "4.1  결정변수와 동역학"),
    ("eq", r"$q_k=[\,p_k,\ \theta_k,\ \alpha_k\,]\in\mathrm{R}^{14}$   "
           r"베이스 위치 $p$, 자세 회전벡터 $\theta$, 팔관절 $\alpha$ (8개)"),
    ("eq", r"$v_k=[\,\dot{p}_k,\ \omega_k,\ \dot{\alpha}_k\,]\in\mathrm{R}^{14}$,   "
           r"$u_k\in\mathrm{R}^{14}$ 가속도,   $b_k\in\mathrm{R}^{3}$ 박스 위치"),
    ("eq", r"동역학: $\ v_{k+1}=v_k+u_k\,\Delta t,\qquad q_{k+1}=q_k+v_{k+1}\,\Delta t$  "
           r"(semi-implicit Euler)"),
    ("body", "팔은 팔당 4관절. dof1(연속 관절)은 두 팔에 반대부호로 주어 대칭으로 아래로 뻗는다. 베이스 "
             "렌치 / 로터 allocation은 하류 gRITE에 위임한다(가속도 레벨 transcription)."),

    ("h3", "4.2  베이스 프레임 파지와 접촉"),
    ("body", "(핵심) 스퀴즈 축을 월드 x가 아니라 베이스 x로 둔다. 베이스가 기울면 월드 패드간격이 흔들려 "
             "좌우 수직력이 발산하지만(목적함수 ~1e7로 폭발), 베이스 프레임 파지(팔만의 FK, 베이스 = 단위)에서는 "
             "자세 불변이라 박스가 베이스와 강체로 함께 기운다."),
    ("eq", r"$b^{world}=p+R(\theta)\,\cdot\,\mathrm{mid}_{arm}$    (박스 중심이 그립 중점에 강체로 부착)"),
    ("eq", r"$\varphi_L=(b_x-p^{L}_{x})-h,\quad \varphi_R=(p^{R}_{x}-b_x)-h$   (베이스 프레임 가상 침투)"),
    ("eq", r"$\lambda(\varphi)=\frac{1}{2} k_c\left(\sqrt{\varphi^{2}+\epsilon^{2}}-\varphi\right)$   "
           r"(분리$\to$0, 침투$\to -k_c\varphi$; 매끄러움)"),

    ("h3", "4.3  파지 제약"),
    ("bul", "박스 위치: approach / grasp에서 b = pick(책상), release에서 b = place(선반), "
            "grasp / transport / release에서 b = b_world(그립 중점에 강체)."),
    ("bul", "패드를 박스 양면 중심에 핀(베이스 프레임 y, z): p^L_y = p^R_y = b_y, p^L_z = p^R_z = b_z "
            "(양면 중심을 평행하게; x 방향은 스퀴즈가 누름). -> 관찰된 '면 중심 파지'가 계획에서 정확히 성립."),
    ("bul", "평행 그리퍼 유지(패드면 각도 = 0). 좌우 대칭은 approach / grasp에서 강제하고, 창문 통과 중에는 "
            "비대칭 reach를 허용하되 양 팔 모두 평행 jaw는 유지(틈을 비스듬히 지나려면 비대칭 reach 필요)."),
    ("bul", "slip-aware 스퀴즈(transport): lambda_L, lambda_R >= f_set,k. 마찰원뿔이 중력 + 운반가속에 대해 "
            "박스를 항상 잡도록 f_set을 계획된 박스 수직가속으로 정한다."),

    ("h3", "4.4  창문 keep-out  (핵심 기여)"),
    ("body", "창문 통과는 HARD 제약 대신 SOFT 벌점으로 부과한다(하드 비볼록 제약은 IPOPT 승수가 발산). "
             "박스 표면 / 팔 캡슐 / 베이스를 여러 점으로 샘플해, 각 점이 (a) 벽 개구부 안에 있고 "
             "(b) sash 밖에 있도록 벌점한다. 박스 x가 창문 근처(near_win, box_x in [-2.05, 0.6])인 knot에만 "
             "적용한다(원거리에서 LSE 지수폭주 방지)."),
    ("body", "벽 벌점 win_wall - 점 p가 벽 x-슬랩 안이면 개구부 안에 있어야 한다:"),
    ("eq", r"$s_x=\sigma(p_x;x_{lo},x_{hi})$ (x-슬랩 indicator, 폭 $sw{=}0.04$),  "
           r"$m_y,m_z=\mathrm{smin}$(개구부 y,z 여유)"),
    ("eq", r"$\mathrm{win\_wall}=s_x\cdot\mathrm{viol}(\mathrm{smin}(m_y,m_z))$   "
           r"(슬랩 안 AND 개구부 밖 $\to$ 벌점)"),
    ("body", "[수정한 핵심 버그] 이전에는 win_wall = smin(m_y,m_z) + 8(1 - s_x)인 big-M 꼴이었다. 벽 슬랩 "
             "두께(0.10 m)와 indicator 평활폭(0.10 m)이 같아 s_x가 ~0.6까지밖에 못 올라, 8(1 - s_x) ~ 3 m "
             "항이 실제 (미터 미만) 위반을 가려버렸다. 그 결과 개구부 '아래로 완전히 내려간' 박스가 "
             "'+3 m 여유 = 통과'로 읽혀 박스가 벽을 -10 cm 파고들었다. 이를 곱셈형(s_x * viol)으로 바꾸고 "
             "s_x를 sharp(sw = 0.04)하게 하여 위반이 정확히 벌점되게 고쳤다(같은 자세에서 벌점 0 -> ~35000)."),
    ("body", "sash 벌점 win_sash - 점이 기울어진 sash 박스 '밖'에 있도록, 세 면거리(sash의 u, v, n 축)의 "
             "매끄러운 최대(LSE)를 쓴다:"),
    ("eq", r"$\mathrm{win\_sash}=\frac{1}{\beta}\log(e^{\beta d_u}+e^{\beta d_v}+e^{\beta d_n}),\ \ "
           r"d_u=|d\cdot u|-(h_u+r)$   ($>0$이면 밖)"),
    ("eq", r"$\mathrm{viol}(c)=(\frac{1}{2}(\sqrt{c^2+\delta}-c))^2\approx\max(0,-c)^2$   (침투$^2$)"),
    ("body", "몸체 샘플: 베이스 1점(r = 0.18), 팔 링크 세그먼트당 7점(r = 0.06), 박스는 3x3x3 = 27점 격자 "
             "(모서리 + 변 + 면중심 + 중심, r = 0.02). 8모서리만 쓰면 박스 x폭(0.106) ~ 벽두께(0.10)라 "
             "모서리가 슬랩 밖으로 빠져 사각이 생기므로 격자로 보강했다. 총 벌점 = w_win * sum_point "
             "(win_wall + viol(win_sash)), w_win = 60000."),
    ("bul", "베이스 z 상한: p_z <= 1.6 (home). 없으면 soft 벌점이 유한한 4 m 벽을 '넘어' 날아가버린다(통과 대신)."),

    ("h3", "4.5  비용 함수"),
    ("eq", r"$J_{tilt}=w_{tilt}(q_x^2+q_y^2)$,   $q=\exp(\theta)$의 허수부 $=\sin^2(\mathrm{tilt}/2)$ "
           r"$=$ 베이스 z축 off-vertical (yaw 불변). $w_{tilt}{=}40$."),
    ("eq", r"$J_{yawrate}=w_{yr}\,(\dot{\theta}_{z,k})^2=w_{yr}\,v_k[5]^2$,   $w_{yr}{=}10$  (yaw '속도' 벌점)"),
    ("body", "[yaw 비용 - 사용자 요청] 'yaw가 회전하는 양'을 벌점한다. yaw '값'(q_z^2)을 벌하면 창문 통과에 "
             "필요한 결합된 틸트까지 끌어내려 박스가 클립되지만, yaw '속도'(d theta_z / dt = v[5])를 벌하면 "
             "필요한 알짜 회전은 유지한 채 불필요한 왕복 흔들림만 제거한다. 창문 keep-out가 통과에 필요한 "
             "yaw를 강제하므로 전역(gated 아님)으로 줘도 클립으로 도망가지 않는다. 결과: 전체 yaw 이동량 "
             "441도 -> 80도(단조 ramp, 왕복 흔들림 0)."),
    ("eq", r"$J_{reg}=w_a\|u_k\|^2+w_v\|v_k\|^2$ (노력),  "
           r"$J_{align}=w_{al}((b_x{-}p_x)^2+(b_y{-}p_y)^2)$ (박스를 베이스 아래로)"),
    ("body", "J_align은 창문 근처에서는 약화한다(박스가 옆으로 비켜 틈을 지나야 하므로). 샘플러 약한 추종 "
             "(w_trk)으로 베이스 위치 + 팔을 샘플러 경로 쪽으로 끌어(자세는 추종 안 함, w_trk_th = 0) "
             "fly-over / 나쁜 minima 대신 충돌 없는 위상에 머물게 한다(사용자 선호에 따라 점차 "
             "cost / constraint로 대체 예정)."),

    ("h3", "4.6  solver / warm-start"),
    ("body", "IPOPT(CasADi Opti)로 풀며, 샘플러 경로의 일관된 운동학 seed(위치 / 속도 / 가속)로 "
             "warm-start한다(영속도 guess는 metre 스케일 비행에서 restoration 실패). 결과: Optimal, 박스가 "
             "개구부로 올라가 통과(+4.6 cm 여유). 출력 = 박스 / 베이스 / 팔 궤적 + gRITE용 reference "
             "(위치, 속도, 가속, 자세 R, 각속도 omega, 각가속 등)."),
    ("gap", 0.3),

    ("h2", "5.  3단계 - gRITE 기하-강건 제어기"),
    ("body", "reference를 실제 물리에서 추종하는 제어기는 단순 기하 PID가 아니라 gRITE(geometric + RISE) "
             "이다: 기하 SE(3) 추종 + RISE 강건항(연속 적분 + tanh 부호 -> 비대칭 외란의 점근적 제거) + "
             "모델 피드포워드(자이로 omega x J omega, 관성, 중력 모멘트). 실제 비행 C++ 제어기를 이식."),
    ("eq", r"$e_p=p-p_d,\ \ e_v=v-v_d,\ \ \Psi=R^{T}R_d,\ \ "
           r"e_R=\mathrm{vee}(\frac{1}{2}(\Psi^{T}-\Psi)),\ \ e_\omega=\omega_b-\Psi\,\omega_d$"),
    ("eq", r"필터오차: $\ e_{t1}=e_v+\Lambda_{t1}e_p,\qquad e_{r1}=e_\omega+\Lambda_{r1}e_R$"),
    ("body", "병진 힘(베이스 프레임): 명목 + RISE 적분항"),
    ("eq", r"$f_n=m\,R^{T}(a_d-K_{tp}e_p-K_{td}e_v+g\,e_3)$"),
    ("eq", r"$f_{ri}\leftarrow f_{ri}+\Delta t((K_{ti}{+}\rho_t)e_{t1}+\Lambda_{t2}\tanh(\Theta_t e_{t1})),\ "
           r"\ f_r=-R^{T}((K_{ti}{+}\rho_t)e_{t1}+f_{ri})$"),
    ("body", "회전 토크: 자이로 / 관성 피드포워드 + 기하 PD + 중력모멘트 + RISE"),
    ("eq", r"$\tau_n=\hat{\omega}(J\omega)-J(\hat{\omega}\,\Psi\omega_d-\Psi\dot{\omega}_d)"
           r"-J K_{rp}e_R-J K_{rd}e_\omega+m g\,(c_{b}\times R^{T}e_3)$"),
    ("body", "c_b는 (로봇만의) CoM 오프셋이다. 운반하는 박스의 모멘트는 RISE 항이 강건하게 제거하므로 "
             "모델에 넣지 않는다(제어기가 payload 모델을 요구하지 않는다는 더 강한 주장). 박스 질량은 "
             "추력 크기에만 더한다. gain(튜닝): K_tp = [22,22,36], K_td = [10,10,14], K_ti = [8,8,14]; "
             "K_rp = [520,520,150], K_rd = [52,52,26], K_ri = [18,18,8] (회전 K_rp / K_rd는 J-스케일된 큰 "
             "값, RISE K_ti / K_ri는 직접 렌치 게인)."),
    ("gap", 0.3),

    ("h2", "6.  4단계 - 물리 시뮬레이션 (IsaacSim)"),
    ("body", "IsaacSim / IsaacLab에서 gRITE 렌치를 베이스에 직접 인가한다(로터 allocation은 추상화). 박스는 "
             "동적 강체(RigidObject, 1 kg)로 책상 위에 놓여 있다가 패드 마찰로 잡혀 운반된다(가짜 attach "
             "없음; 따라서 잡히기 전에는 책상 위에 가만히 있고 스스로 돌지 않는다). 장면: 책상과 랙(x = -4)은 "
             "정적 콜라이더, 창문 USD에는 콜라이더가 없어 계획기의 keep-out 5개를 반투명 솔리드 콜라이더로 "
             "스폰한다. 15 초 horizon을 반복(loop) 재생한다."),
    ("gap", 0.3),

    ("h2", "7.  결과 (현재 시뮬레이션)"),
    ("bul", "박스가 개구부를 +4.6 cm 여유로 통과(관통 knot 0개; coal 검증 149개 중 145개 knot clear)."),
    ("bul", "yaw 전체 이동량 441도 -> 80도(단조 회전, 왕복 흔들림 제거); off-vertical 틸트 최대 27.6도."),
    ("bul", "파지: 계획은 박스 양면 중심을 정확히(베이스 프레임 y, z = 0) 평행하게 잡는다. 동적 sim의 마찰 "
            "파지는 강한 운반에서 ~10 cm 미끄러질 수 있다(별도 과제: 스퀴즈 강화 또는 운반 감속)."),
    ("bul", "남은 과제: 베이스(드론 동체)가 가장 좁은 순간 프레임을 ~5 cm 스친다(동체를 단일 구로 과소모델 "
            "-> 박스와 같은 방식의 조밀 샘플로 보강 예정)."),
]

PAGE_W, PAGE_H = 8.27, 11.69      # A4 inches
LEFT, RIGHT, TOP, BOT = 0.085, 0.93, 0.945, 0.06
LH = 0.0155


def dwidth(s):
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
    path = os.path.join(out_dir, "window_method_ko.pdf")
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
                put(text, 0.5, 11, weight="bold", ha="center", color="#333333"); y[0] -= 0.044
            elif kind == "h2":
                if y[0] < BOT + 0.10:
                    newpage()
                y[0] -= 0.008
                put(text, LEFT, 13.5, weight="bold", color="#11337a"); y[0] -= 0.030
            elif kind == "h3":
                if y[0] < BOT + 0.08:
                    newpage()
                y[0] -= 0.004
                put(text, LEFT, 11, weight="bold", color="#3a5a1a"); y[0] -= 0.024
            elif kind == "eq":
                eqlh = 0.031 if ("\\frac" in text or "\\sqrt" in text or "\\tanh" in text) else 0.022
                if y[0] < BOT + eqlh:
                    newpage()
                put("    " + text, LEFT, 11, color="#7a1111"); y[0] -= eqlh + 0.005
            else:  # body, bul
                lines = kwrap(text, 92 if kind == "bul" else 96)
                indent = LEFT + (0.018 if kind == "bul" else 0.0)
                for i, ln in enumerate(lines):
                    if y[0] < BOT:
                        newpage()
                    put(("•  " + ln if i == 0 else "   " + ln) if kind == "bul" else ln, indent, 9.7)
                    y[0] -= LH
                y[0] -= 0.004
        pdf.savefig(fig); plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
