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
             "책상 위 박스로 날아가 마찰로 잡고, 운반해, 랙 맨 위 선반에 놓는 동작이다. "
             "Transcription은 가속도 레벨이고 explicit(semi-implicit) Euler 적분을 쓰며, "
             "베이스 렌치/로터 allocation은 하류의 gRITE 제어기에 위임한다(결정 D5)."),
    ("gap", 0.4),

    ("h2", "1.  설정, 좌표계, horizon, phase FSM"),
    ("body", "모든 양은 HOME 좌표계 기준이고, 원점은 드론 스폰 지점(월드 (0,0,1.5) m)이다. "
             "베이스 자세는 회전벡터 theta in R^3(지수좌표)로 매개화하고 유클리드 공간에서 "
             "적분한 뒤, forward kinematics 입력에만 단위 쿼터니언으로 변환한다(단위노름 manifold "
             "등식을 피해 IPOPT 발산을 막음)."),
    ("body", "task는 박스의 두 점 pick과 place로 주어진다. 베이스가 수평일 때 end-effector(EE) "
             "중점은 베이스에서 고정 오프셋 r_off에 놓인다. 팔이 아래로 뻗어 잡으므로(5c) r_off는 "
             "아래(-z) 성분을 가지며, 따라서 base_grasp = pick - r_off는 박스보다 위, 즉 책상 "
             "표면에서 약 0.33 m 떠 있게 된다(r_off는 grasp 형상의 FK로 한 번 계산)."),
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
             "Pinocchio 모델은 USD에서 만들고 Isaac과 0.00 mm 일치 검증함."),
    ("gap", 0.4),

    ("h2", "3.  접촉 모델 (결정변수가 아니라 유도량)"),
    ("body", "두 패드가 박스를 월드 x축으로 누른다(팔이 아래로 뻗어도 패드 면은 평행 그리퍼 제약 "
             "5c로 월드 x에 수직 유지되어 스퀴즈 축은 여전히 월드 x). EE 위치 (p^L, p^R)는 FK로 "
             "구한다. 가상 침투량 phi(양수=분리, 음수=침투)로 매끄러운 수직력 lambda를 정의한다 "
             "(결정 D2/D3: 침투에서 유도, 상보성(complementarity) 없음, 접촉력 결정변수 없음):"),
    ("eq", r"$\varphi_L=(b_x-p^{L}_{x})-h,\qquad \varphi_R=(p^{R}_{x}-b_x)-h$"),
    ("eq", r"$\lambda(\varphi)=\frac{1}{2}\,k_c\left(\sqrt{\varphi^{2}+\epsilon^{2}}-\varphi\right)$"),
    ("body", "분리 시(phi>0) lambda->0, 침투 시(phi<0) lambda->-k_c*phi이며 phi=0에서 도함수가 "
             "연속이다. 파라미터: k_c = 2000 N/m, epsilon = 1e-4 m, 반폭 h = 0.0525 m (10.5 cm 박스)."),
    ("gap", 0.4),

    ("h2", "4.  비용함수  J = sum_k ( phase별 항 )  +  정규화"),
    ("body", "각 knot은 phase별 항과, 모든 phase 공통인 자세/정규화 항을 더한다. lambda_L, "
             "lambda_R은 위의 패드 수직력이고, p_ref, b_guess, alpha*는 6장의 reference 경로다."),
    ("eq", r"$J_{base}=w_{base}\,\| p_k-p_{ref,k}\|^{2}$"
           r"   (approach: 비행 ; grasp/release: hover 유지)"),
    ("eq", r"$J_{box}=w_{box}\,\| b_k-b_{guess,k}\|^{2}$"
           r"   (transport: 부드러운 up-and-over guide 추종)"),
    ("eq", r"$J_{f}=w_{f}[(\lambda_L-F^{*})^{2}+(\lambda_R-F^{*})^{2}]$"
           r"   스퀴즈를 $F^{*}$로 (grasp/transport 22 N, release 2 N)"),
    ("eq", r"$J_{sym}=w_{sym}\,(\lambda_L-\lambda_R)^{2}$    좌우 스퀴즈 대칭"),
    ("eq", r"$J_{hold}=w_{hold}\,\| \alpha_k-\alpha^{*}\|^{2}$"
           r"    팔 형상 유지 (표적 $\alpha^{*}$는 아래 설명)"),
    ("eq", r"$J_{att}=w_{att}\,\|\theta_k\|^{2}$    베이스 수평 유지 (모든 phase)"),
    ("eq", r"$J_{reg}=w_{a}\,\| u_k\|^{2}+w_{v}\,\| v_k\|^{2}$"
           r"    가속도 / 속도 최소화 (모든 phase)"),
    ("body", "J_hold의 표적 alpha*: grasp/transport에서는 팔 전체를 alpha_grasp(reach-down 닫힘)로 "
             "추종한다. 단 approach에서는 그리퍼를 연 상태(dof2,3,4)만 추종하고 dof1은 자유로 둔다 "
             "(팔 내림을 강제하지 않기 위함, 5c). 가중치: w_base = 50, w_box = 200, w_f = 1, "
             "w_sym = 2, w_hold = 2, w_att = 30, w_a = 1e-3, w_v = 1e-2. 종단 w_att||theta_N||^2 추가."),
    ("body", "Prescribe vs discovered(중요): (i) transport의 up-and-over 모양은 soft guide b_guess가 "
             "정하고, 5b의 keep-out은 clearance를 보장하는 HARD 제약이다(모양=guide, 회피=제약). "
             "(ii) 반대로 grasp 때 팔이 내려가는 동작은 prescribe하지 않는다. base 클리어런스 제약 + "
             "박스를 잡아야 한다는 조건만으로 팔 내림이 유일한 해가 되어, 최적화가 스스로 발견한다(5c)."),
    ("gap", 0.4),

    ("h2", "5.  제약조건"),
    ("bul", "초기조건:  q_0 = [ home, theta = 0, alpha_start ],  v_0 = 0.  alpha_start는 팔이 "
            "안 내려간 수평 자세(dof1 = 0, 그리퍼는 열림)."),
    ("eq", r"동역학 $(k=0..N\!-\!1)$:   $v_{k+1}=v_k+u_k\,\Delta t,\quad "
           r"q_{k+1}=q_k+v_{k+1}\,\Delta t$"),
    ("bul", "phase별 박스 위치 (no-slip carried-box 모델, 결정 D6):"),
    ("eq", r"approach, grasp:  $b_k=\mathrm{pick}$       (박스가 책상 지지대에 놓임)"),
    ("eq", r"transport:  $b_k=\frac{1}{2}(p^{L}_k+p^{R}_k)$   (EE 중점에서 강체로 운반)"),
    ("eq", r"release:  $b_k=\mathrm{place}$;     종단:  $b_N=\mathrm{place}$"),
    ("bul", "수직 강하 (approach 마지막 ~45%): 박스 바로 위에서 베이스 수평위치를 고정한다."),
    ("eq", r"$p_{x,k}=p^{g}_{x},\qquad p_{y,k}=p^{g}_{y}$"
           r"   ($p^{g}$ = base_grasp; 열린 패드가 옆으로 쓸지 않고 수직 강하)"),
    ("bul", "slip-aware 스퀴즈 (transport 구간) : 헤드라인 법칙, HARD 부등식:"),
    ("eq", r"$\lambda_L\geq f_{set,k},\qquad \lambda_R\geq f_{set,k}$"),
    ("eq", r"$f_{set,k}=\max(\frac{m_o\,(g+a_{z,k})}{2\mu}+\mathrm{margin},\ \mathrm{floor})$"),
    ("body", "a_{z,k}는 계획된 박스 수직 가속도(reference 박스 z의 2차 차분). m_o = 0.8 kg, "
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
             "잔차만 랙 중앙에서 완화하므로, 박스는 랙 접근에서 높이 올라갔다가 slot으로 선반에 내려온다."),
    ("gap", 0.4),

    ("h2", "5c.  Reach-down 파지: 베이스 클리어런스 · 평행 그리퍼 · 자연스러운 팔 내림"),
    ("body", "베이스가 책상/랙 표면에 너무 가까우면 지상효과가 크고 작은 오차로도 충돌한다. 그래서 "
             "드론 베이스(팔 제외)는 표면 위로 최소 c_lr 이상 떠 있어야 한다. 5b와 같은 indicator를 "
             "쓰되 박스 반높이 없이 표면 자체(z^surf = 장애물 top)까지 잰다:"),
    ("eq", r"$p_z+M\,(1-I_{desk}(p))\ \geq\ z^{surf}_{desk}+c_{lr}$    (책상 위 $c_{lr}$ 이상)"),
    ("eq", r"$p_z+M\,(1-I_{rack}(p))\ \geq\ z^{surf}_{shelf}+c_{lr}$   (랙 선반 위 $c_{lr}$ 이상)"),
    ("body", "c_lr = 0.30 m. 박스는 표면(책상/선반)에 있으므로, 베이스가 위에 떠 있으면 팔이 아래로 "
             "뻗어야 박스에 닿는다. 관절1(dof1)을 두 팔에 반대부호로 주면 두 팔이 대칭으로 약 0.27 m "
             "내려가는데, 패드의 x(파지 폭)와 월드 x 면 법선은 보존하고 수직 위치만 낮춘다. 그래서 "
             "r_off의 z = -0.27 m, base_grasp는 책상 위 약 0.33 m에 뜬다."),
    ("body", "자연스러운 팔 내림(핵심): dof1에 내림 reference를 주지 않는다. 'base는 표면에서 c_lr "
             "이상 떨어진다'는 제약과 '박스를 잡아야 한다'는 조건이 함께 작용하면 팔이 내려가는 것이 "
             "유일한 해가 되어, 최적화가 dof1을 스스로 내린다(수평 dof1 = 0에서 grasp의 약 0.68까지, "
             "가속도 정규화로 매끄럽게). approach 비용은 그리퍼 열림(dof2-4)만 잡아두고 dof1은 자유다."),
    ("bul", "평행 그리퍼 (모든 knot, k >= 1; k = 0은 IC로 고정): 패드 면 각도는 팔당 d2-d3-d4의 "
            "선형식이고 pregrasp와 grasp 모두에서 0이라, 이를 0으로 강제한다."),
    ("eq", r"$\alpha^{d2}_k-\alpha^{d3}_k-\alpha^{d4}_k=0$    (왼팔, 오른팔 각각)"),
    ("body", "이로써 닫힘이 순수 스퀴즈가 된다: 간격만 줄고 패드는 회전하지 않아 박스를 밀어내지 "
             "않는다. dof1은 이 식을 보존하므로, 팔을 내려도 면은 계속 월드 x에 수직(평행)이다."),
    ("gap", 0.4),

    ("h2", "6.  Reference 경로, 초기 guess, solver"),
    ("body", "p_ref(approach)는 over-then-down이다: 시작 높이에서 박스 바로 위까지 수평 이동한 뒤 "
             "수직 강하한다(베이스는 박스 높이로 내려오지 않고 base_grasp = 표면 위 0.33 m에 머물며, "
             "팔이 내려가 박스를 잡는다). grasp/release에서는 grasp/place hover. b_guess는 부드러운 "
             "시간-기반 up-and-over(pick에서 place_z + 0.6 m까지 수직 상승, 고고도 이동, transport "
             "마지막 ~25%에서 place로 수직 하강)."),
    ("body", "팔 형상: alpha_start(수평, dof1 = 0; IC) -> alpha_pre(reach-down 열림) / "
             "alpha_grasp(reach-down 닫힘). pregrasp와 grasp는 reach-down(dof1)과 reach(y)가 같고 "
             "x(간격)만 다르며, d2-d3-d4 = 0이라 닫힘이 순수 평행 스퀴즈다(dof2-4는 패드 메시에서 도출)."),
    ("body", "초기 guess: 전체 운동학 시드(reference와 일관된 위치/속도/가속도). 영속도 guess는 "
             "metre 스케일 비행에서 IPOPT restoration에 실패한다. Solver: IPOPT via CasADi Opti, "
             "max_iter 5000, tol 1e-4, adaptive mu. 출력: 박스/베이스/팔 궤적과 30차원 gRITE "
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
