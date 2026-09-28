# app.py ― パイプライン型ADC 学習アプリ（Streamlit）
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

st.set_page_config(page_title="Pipelined ADC 入門", layout="wide")

VREF = 2.0  # 基準電圧（固定）。入力レンジは 0 ～ Vref
KW = ["全体構造", "残差・残差増幅器 (MDAC)", "パイプライン処理", "時差・誤差補正 (DEC)"]
TH_LO, TH_HI = 0.75, 1.25    # 1.5bit Sub-ADC のしきい値
LAST_LO, LAST_HI = 0.5, 1.5  # 最終段（MDACなし）のしきい値


# ===================== 計算部 =====================
def weight(k, n):
    """Stage k の出力 D_k がコードに加算されるときの重み（DECの1bitずらし加算）"""
    return 1 if k == n else 2 ** (n - 1 - k)


def run_adc(vin, n, off=0.0):
    """1.5bit/段モデル。D=0,1,2。残差 Vres = 2*(Vin - D*Vref/4) = 2Vin - D
    off: Sub-ADC比較器のオフセット[V]（全段のしきい値を同量ずらす）"""
    v = min(max(vin, 0.0), VREF)
    rows, m = [], 0
    for k in range(1, n + 1):
        last = k == n
        lo, hi = (LAST_LO, LAST_HI) if last else (TH_LO, TH_HI)
        d = 0 if v < lo + off else (1 if v < hi + off else 2)
        vdac = d * VREF / 4
        vsub = v - vdac
        vres = np.nan if last else 2 * vsub
        rows.append(dict(stage=k, vin=v, d=d, vdac=vdac, vsub=vsub, vres=vres))
        m += d * weight(k, n)
        v = vres
    return rows, min(m, 2 ** n - 1), m


def transfer(x, off=0.0):
    d = np.where(x < TH_LO + off, 0, np.where(x < TH_HI + off, 1, 2))
    return 2 * x - d


# ===================== 全体回路ブロック図 =====================
def draw_block(n, focus):
    ON, OFF, ALL = "#ffd54f", "#e3eaf2", "#c8e6c9"
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.axis("off")
    W, G, X0 = 2.0, 0.4, 1.9
    xe = X0 + n * (W + G)
    ax.set_xlim(-0.8, xe + 3.2)
    ax.set_ylim(-0.4, 4.4)
    ov, mdac, pipe, dec = (focus == KW[i] for i in range(4))

    def box(x, y, w, h, txt, on):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                    fc=ALL if ov else (ON if on else OFF), ec="#345", lw=1.5))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)

    def arr(p, q):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="->", lw=1.4))

    ax.text(-0.75, 3.15, "Vin", fontsize=11)
    arr((-0.3, 3.0), (0.2, 3.0))
    box(0.2, 2.4, 1.2, 1.2, "S/H\n(CLK)", pipe)
    arr((1.4, 3.0), (X0, 3.0))
    for k in range(1, n + 1):
        x = X0 + (k - 1) * (W + G)
        last = k == n
        txt = (f"Stage {k}\nSub-ADC + DAC\n-> Sum -> x2 (MDAC)" if not last
               else f"Stage {k}\nSub-ADC only\n(last stage)")
        box(x, 2.4, W, 1.2, txt, pipe or (mdac and not last))
        if not last:
            arr((x + W, 3.0), (x + W + G, 3.0))
            ax.text(x + W + G / 2, 3.12, "Vres", ha="center", fontsize=8)
        arr((x + W / 2, 2.4), (x + W / 2, 1.5))
        ax.text(x + W / 2 + 0.05, 1.85, f"D{k}", fontsize=9)
        box(x + 0.1, 0.7, W - 0.2, 0.8, f"Shift Reg\n({n - k} CLK delay)", dec)
        ax.plot([x + W / 2] * 2, [0.7, 0.3], c="#345", lw=1.4)
    ax.plot([X0 + W / 2, xe], [0.3, 0.3], c="#345", lw=1.4)
    arr((xe, 0.3), (xe + 0.3, 0.3))
    box(xe + 0.3, -0.1, 1.9, 1.2, "DEC\n(Digital Error\nCorrection)", dec)
    arr((xe + 2.2, 0.5), (xe + 2.7, 0.5))
    ax.text(xe + 2.75, 0.5, "Dout\n(N bit)", va="center", fontsize=10)
    label = ["Overview", "MDAC / Residue", "Pipeline (S/H + all stages)", "Shift Reg + DEC"][KW.index(focus)]
    ax.set_title(f"Pipelined ADC block diagram (N = {n})   highlight: {label}", fontsize=11)
    return fig


# ===================== 0. 全体構造 =====================
def view_overview(rows, code, n):
    st.subheader("全体構造")
    st.markdown(
        "サイドバーでキーワードを選ぶと、対応するブロックがハイライトされます。\n\n"
        "1. **残差・MDAC**：各段が粗く判定し、その分を引いた**残差を2倍**して次段へ渡す。\n"
        "2. **パイプライン処理**：各段のS/Hが値を保持し、**全段が別サンプルを同時に処理**する。\n"
        "3. **時差・誤差補正 (DEC)**：段ごとに確定時刻が違うビットを**シフトレジスタで揃え**、"
        "**1bitずらして加算**して比較器誤差を補正しながら1つのコードにする。")
    df = pd.DataFrame(rows).rename(columns={
        "stage": "段", "vin": "Vin[V]", "d": "D", "vdac": "DAC[V]",
        "vsub": "Vin-DAC[V]", "vres": "Vout(残差×2)[V]"}).set_index("段")
    st.dataframe(df.round(3))
    st.caption("最終段はMDACを持たず、Sub-ADCの判定のみを出力します（Vout は NaN）。")


# ===================== 1. 残差・MDAC =====================
def view_mdac(rows, n):
    st.subheader("残差・残差増幅器 (MDAC)")
    steps = ["① Vin入力", "② Sub-ADC判定", "③ DAC減算（残差）", "④ ×2増幅（Vout）"]
    c1, c2 = st.columns(2)
    k = c1.slider("解析するステージ（最終段はMDACなし）", 1, n - 1, 1)
    step = c2.radio("計算ステップ", steps, horizontal=True)
    si = steps.index(step)
    s = rows[k - 1]
    cmp_txt = ("Vin < 0.75 V" if s["d"] == 0 else
               "0.75 V ≤ Vin < 1.25 V" if s["d"] == 1 else "Vin ≥ 1.25 V")
    msgs = [
        f"① 入力：Stage {k} の Vin = **{s['vin']:.3f} V**",
        f"② Sub-ADC（しきい値 0.75 V / 1.25 V）：{cmp_txt} → **D{k} = {s['d']}**",
        f"③ DAC出力 = D × Vref/4 = {s['vdac']:.3f} V、減算：Vin − DAC = **{s['vsub']:.3f} V**（残差）",
        f"④ MDACで2倍増幅：Vout = 2 × {s['vsub']:.3f} = **{s['vres']:.3f} V** → 次段のVinへ",
    ]
    for i in range(si):
        st.caption(msgs[i])
    st.success(msgs[si])

    x = np.linspace(0, VREF, 800)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(x, transfer(x), c="#1565c0", label="Vout = 2Vin - D*Vref/2")
    for t in (TH_LO, TH_HI):
        a1.axvline(t, c="r", ls=":", lw=1)
    a1.plot([], [], c="r", ls=":", label="Sub-ADC thresholds")
    ys = [0, 0, s["vsub"], s["vres"]][si]
    a1.plot([s["vin"]], [ys], "o", c="orange", ms=11, zorder=5)
    if si >= 2:
        a1.plot([s["vin"]] * 2, [0, ys], c="orange", lw=1)
    a1.set(xlabel="Vin [V]", ylabel="Voltage [V]", xlim=(0, 2), ylim=(-0.1, 2.1),
           title=f"Stage {k}: sawtooth transfer curve (residue plot)")
    a1.legend(fontsize=8, loc="upper left")
    a1.grid(alpha=0.3)

    labels = ["Vin", "DAC", "Vin-DAC", "Vout (x2)"]
    vals = [s["vin"], s["vdac"], s["vsub"], s["vres"]]
    cols = ["#90caf9", "#a5d6a7", "#ffcc80", "#ef9a9a"]
    for i in range(4):
        a2.bar(labels[i], vals[i] if i <= si else 0, color=cols[i])
        if i <= si:
            a2.text(i, vals[i] + 0.03, f"{vals[i]:.2f}", ha="center", fontsize=9)
    a2.set(ylim=(0, 2.2), ylabel="Voltage [V]", title=f"Stage {k}: voltage change")
    a2.axhline(VREF, c="r", ls=":")
    st.pyplot(fig)
    plt.close(fig)

    fig2, ax = plt.subplots(figsize=(12, 2.8))
    idx = list(range(1, n))
    vs = [rows[i - 1]["vres"] for i in idx]
    ax.bar(idx, vs, color=["orange" if i == k else "#90caf9" for i in idx])
    ax.axhline(VREF, c="r", ls=":")
    ax.set(xlabel="Stage", ylabel="Vout [V]", ylim=(0, 2.3), title="Output (residue) of each stage")
    ax.set_xticks(idx)
    for i, v in zip(idx, vs):
        ax.text(i, v + 0.04, f"{v:.2f}", ha="center", fontsize=8)
    st.pyplot(fig2)
    plt.close(fig2)
    st.caption("残差は常に 0 ～ Vref に収まるため、次段は同じ回路で同じ処理を繰り返せます。")


# ===================== 2. パイプライン処理 =====================
def view_pipe(vin, n):
    st.subheader("パイプライン処理")
    ns, names = 4, ["A", "B", "C", "D"]
    vins = [min(max(v, 0), VREF) for v in
            [vin, vin * 0.6 + 0.3, (vin + 0.9) % 2, 2 - vin * 0.5]]
    res = [run_adc(v, n)[0] for v in vins]
    t = st.slider("クロックサイクル t", 1, n + ns - 1, 1)
    data = {}
    for c in range(1, t + 1):
        row = {"S/H(入力)": f"{names[c-1]} ({vins[c-1]:.2f}V)" if c <= ns else "—"}
        for k in range(1, n + 1):
            s = c - k + 1
            row[f"Stage{k}"] = f"{names[s-1]}: D{k}={res[s-1][k-1]['d']}" if 1 <= s <= ns else "—"
        s = c - n + 1
        row["変換完了"] = f"{names[s-1]} 完了" if 1 <= s <= ns else "—"
        data[f"CLK {c}"] = row
    df = pd.DataFrame(data).T
    st.dataframe(df.style.apply(
        lambda r: ["background-color:#ffd54f"] * len(r) if r.name == f"CLK {t}" else [""] * len(r),
        axis=1))
    a, b, c = st.columns(3)
    a.metric("レイテンシ", f"{n} クロック")
    b.metric("スループット", "1 サンプル / クロック")
    c.metric("同時処理サンプル数", f"最大 {n}")
    st.markdown(
        "- **レイテンシ**：入力されてから結果が出るまでの遅延。段数Nに比例し、**Nクロック**かかります。\n"
        "- **スループット**：各段のS/H（レジスタ）が値を保持するので、全段が別サンプルを並列処理でき、"
        "**定常状態では毎クロック1サンプル**を出力できます。\n"
        "- 段数を増やすと分解能は上がりますが、増えるのは**レイテンシだけ**でスループットは変わりません。")


# ===================== 3. 時差・誤差補正 (DEC) =====================
def view_dec(vin, n):
    st.subheader("時差・誤差補正 (DEC)")
    c1, c2 = st.columns(2)
    t = c1.slider("クロック（サンプルAがStage1に入ってからの経過）", 1, n, n)
    off = c2.slider("比較器オフセット [V]（全段のSub-ADCしきい値をずらす）", -0.4, 0.4, 0.0, 0.05)
    rows, code, raw = run_adc(vin, n, off)
    _, ideal, _ = run_adc(vin, n, 0.0)

    st.markdown("##### ① タイムアライメント（Stage k のビットはクロック k で確定 → N−k クロック遅延）")
    cell = {}
    for k in range(1, n + 1):
        nm = f"D{k}" + ("(MSB側)" if k == 1 else "(LSB側)" if k == n else "")
        d = rows[k - 1]["d"]
        if t < k:
            cell[nm] = "— (未確定)"
        else:
            pos, dly = t - k, n - k
            cell[nm] = f"{d}  ✔整列" if pos >= dly else f"{d}  (遅延中 {pos}/{dly})"
    st.dataframe(pd.DataFrame([cell], index=[f"CLK {t}"]))
    st.dataframe(pd.DataFrame({
        "ビット": [f"D{k}" for k in range(1, n + 1)],
        "確定クロック": list(range(1, n + 1)),
        "遅延段数(N-k)": [n - k for k in range(1, n + 1)],
        "出力クロック": [n] * n}).set_index("ビット").T)

    st.markdown("##### ② DEC：各段の出力（D=0,1,2）を **1bitずつずらして加算**")
    if t < n:
        st.info(f"クロック {n} で全ビットが揃い、加算が行われます。スライダーを進めてください。")
    else:
        w = n + 1
        lines = []
        for r in rows:
            k = r["stage"]
            sh = int(np.log2(weight(k, n)))
            lines.append(f"D{k} = {r['d']}   {format(r['d'] << sh, f'0{w}b')}")
        lines.append("-" * (9 + w))
        lines.append(f"合計      {format(raw, f'0{w}b')}  = {raw}"
                     + ("  → 上限にクリップ" if raw != code else ""))
        st.code("\n".join(lines))
        st.caption("隣り合う段のビットが1bit重なって足される（冗長性）ことで、"
                   "前段の判定ミスを後段の残差が打ち消します。最下位2段は同じ重みです。")

        lsb = VREF / 2 ** n
        vrec = code * lsb
        err = min(max(vin, 0), VREF) - vrec
        a, b, c, d = st.columns(4)
        a.metric("Dout (2進)", f"{code:0{n}b}")
        b.metric("Dout (10進)", f"{code} / {2**n - 1}")
        c.metric("再構成電圧", f"{vrec:.3f} V")
        d.metric("量子化誤差", f"{err:+.3f} V", help=f"1 LSB = Vref/2^N = {lsb:.4f} V")
        st.caption(f"1 LSB = {lsb:.4f} V。理想の量子化誤差は ±0.5 LSB（±{lsb/2:.4f} V）以内"
                   "（最上位コード付近はクリップにより最大1 LSB）。")
        if abs(code - ideal) <= 1:
            st.success(f"✅ オフセット {off:+.2f} V があっても、DEC後のコード {code} は理想値 {ideal} と"
                       "ほぼ一致（±1 LSB以内）。冗長性による誤差補正が働いています。")
            st.caption("※ 最終段のしきい値自体のずれにより、±1 LSB程度は残ります。")
        else:
            st.error(f"❌ コード {code} が理想値 {ideal} から大きくずれました。"
                     "残差が次段の入力範囲を超え、補正可能範囲（目安 ±0.25 V ＝ Vref/8）を超えています。")

    x = np.linspace(0, VREF, 800)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(x, transfer(x, off), c="#1565c0")
    ax.axhspan(0, VREF, color="green", alpha=0.08, label="next-stage input range")
    for th in (TH_LO + off, TH_HI + off):
        ax.axvline(th, c="r", ls=":", lw=1)
    ax.set(xlabel="Vin [V]", ylabel="Vout [V]", xlim=(0, 2), ylim=(-0.3, 2.3),
           title=f"Transfer curve with comparator offset = {off:+.2f} V")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("しきい値がずれても残差は 0 ～ Vref に収まる間は、後段が正しく変換し直せます。"
               "ずれが大きくなって残差が緑の範囲外に出ると、補正できません。")


# ===================== メイン =====================
st.title("🔧 パイプライン型AD変換器の動作原理")
st.caption("高専4年 電気・電子・情報系向け ／ 1.5bit/段モデル")

with st.sidebar:
    st.header("フォーカス切替")
    focus = st.radio("表示内容", KW)
    st.header("パラメータ")
    vin = st.slider("アナログ入力電圧 Vin [V]", 0.0, 2.0, 1.37, 0.01)
    n = st.slider("ステージ数 N", 3, 6, 4)
    st.number_input("基準電圧 Vref [V]（固定）", value=VREF, disabled=True)
    rows, code, _ = run_adc(vin, n)
    st.metric("出力コード", f"{code:0{n}b}", f"{code}（10進）")

fig = draw_block(n, focus)
st.pyplot(fig)
plt.close(fig)
st.divider()

if focus == KW[0]:
    view_overview(rows, code, n)
elif focus == KW[1]:
    view_mdac(rows, n)
elif focus == KW[2]:
    view_pipe(vin, n)
else:
    view_dec(vin, n)
# ============================================================================
# 【追加機能】電圧の流れ・情報の流れを、矢印でたどって理解する
#   ※ 既存コードは変更していません。この下のブロックを app.py の末尾に追記します。
# ============================================================================
from matplotlib.patches import Circle, Polygon

V_COL, I_COL, G_COL = "#1565c0", "#d84315", "#b0b8c0"  # 電圧=青 / 情報=赤 / 未通過=灰


def _arrow(ax, p, q, state, kind, label=None, lpos=(0.0, 0.12), head=True, fs=8.5):
    """state: future(未通過) / done(通過済) / active(いま流れている)"""
    col = G_COL if state == "future" else (V_COL if kind == "V" else I_COL)
    lw = {"future": 1.2, "done": 2.4, "active": 4.5}[state]
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(
        arrowstyle="-|>" if head else "-", lw=lw, color=col, mutation_scale=14,
        linestyle="--" if state == "future" else "-", shrinkA=0, shrinkB=0))
    if label and state != "future":
        ax.text((p[0] + q[0]) / 2 + lpos[0], (p[1] + q[1]) / 2 + lpos[1], label,
                ha="center", va="bottom", fontsize=fs, color=col, fontweight="bold")


def _first(steps):
    first = {}
    for i, s in enumerate(steps):
        for e in s["edges"]:
            first.setdefault(e, i)
    return first


def _state(eid, first, idx):
    if eid not in first or first[eid] > idx:
        return "future"
    return "active" if first[eid] == idx else "done"


def _legend(ax, loc="upper right"):
    ax.plot([], [], c=V_COL, lw=3, label="Voltage (analog)")
    ax.plot([], [], c=I_COL, lw=3, label="Information (digital)")
    ax.plot([], [], c=G_COL, lw=1.5, ls="--", label="not yet")
    ax.legend(loc=loc, fontsize=8, ncol=3, frameon=False)


# ---------- ① 全体の流れ：ステップ定義 ----------
def _build_steps(rows, n, vin, code, raw):
    S = []
    S.append(dict(kind="V", edges=["in"], boxes=["sh"], title="Vin → S/H",
                  text=f"アナログ電圧 Vin = **{vin:.3f} V** が S/H に入り、クロックに合わせて**サンプリング**されます。"
                       "電圧はキャパシタの電荷として保持されます。"))
    S.append(dict(kind="V", edges=["sh_s1"], boxes=["sh", "s1"], title="S/H → Stage 1",
                  text=f"保持された **{rows[0]['vin']:.3f} V** が Stage 1 の入力になります。"))
    for k in range(1, n):
        r = rows[k - 1]
        S.append(dict(kind="I", edges=[f"d{k}"], boxes=[f"s{k}"], title=f"Stage {k}：Sub-ADCが判定 → D{k}",
                      text=f"Sub-ADC が Vin = {r['vin']:.3f} V を しきい値 0.75 V / 1.25 V と比較し、"
                           f"**D{k} = {r['d']}**（デジタル情報）を出力。この D{k} は ①DACへ ②シフトレジスタへ の2方向に流れます。"))
        S.append(dict(kind="V", edges=[f"v{k}"], boxes=[f"s{k}", f"s{k+1}"],
                      title=f"Stage {k}：DAC減算 → ×2 → Stage {k+1}へ",
                      text=f"DAC が D{k} から **{r['vdac']:.3f} V** を作り、Σ で Vin − DAC = {r['vin']:.3f} − {r['vdac']:.3f} "
                           f"= {r['vsub']:.3f} V（残差）。MDAC が ×2 して **Vres = {r['vres']:.3f} V** を Stage {k+1} へ渡します。"))
    r = rows[n - 1]
    S.append(dict(kind="I", edges=[f"d{n}"], boxes=[f"s{n}"], title=f"Stage {n}（最終段）：判定のみ",
                  text=f"最終段は Vin = {r['vin']:.3f} V をしきい値 0.5 V / 1.5 V で判定して **D{n} = {r['d']}** を出すだけ。"
                       "MDACはありません。→ これで全ビットが出そろいました。"))
    S.append(dict(kind="I", edges=[f"r{k}" for k in range(1, n + 1)], boxes=[f"r{k}" for k in range(1, n + 1)],
                  title="シフトレジスタで時差揃え",
                  text="Stage k のビットは k クロック目に確定済み。早く確定したビットほど長く待たせ（Stage k は N−k クロック遅延）、"
                       "全ビットを同じクロックに揃えます。"))
    S.append(dict(kind="I", edges=["dec"], boxes=["dec"], title="DEC：1bitずらして加算",
                  text="DEC が各Dを重み（2^(N-1-k)）付きで、**1bitずつ重ねて加算**します。"
                       f"合計 = **{raw}**" + (f"（上限にクリップして {code}）" if raw != code else "") + "。"))
    S.append(dict(kind="I", edges=["out"], boxes=["dec"], title="Dout 出力",
                  text=f"最終出力 **Dout = {code:0{n}b}**（10進 {code}）。ここまでが1サンプル分の電圧・情報の旅です。"))
    return S


def draw_flow(n, rows, vin, code, raw, steps, idx):
    first = _first(steps)
    cur = set(steps[idx]["boxes"])
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.axis("off")
    W, G, X0 = 2.0, 0.4, 1.9
    xe = X0 + n * (W + G)
    ax.set_xlim(-0.8, xe + 3.4)
    ax.set_ylim(-0.4, 4.6)

    def box(bid, x, y, w, h, txt):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", lw=1.5, ec="#345",
                                    fc="#ffd54f" if bid in cur else "#e3eaf2"))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=9)

    def edge(eid, p, q, kind, label=None, lpos=(0, 0.12), head=True):
        _arrow(ax, p, q, _state(eid, first, idx), kind, label, lpos, head)

    ax.text(-0.75, 3.15, "Vin", fontsize=11)
    box("sh", 0.2, 2.4, 1.2, 1.2, "S/H")
    edge("in", (-0.3, 3.0), (0.2, 3.0), "V")
    edge("sh_s1", (1.4, 3.0), (X0, 3.0), "V", f"{rows[0]['vin']:.2f}V", (0, 0.1))
    for k in range(1, n + 1):
        r = rows[k - 1]
        x = X0 + (k - 1) * (W + G)
        last = k == n
        box(f"s{k}", x, 2.4, W, 1.2,
            f"Stage {k}\nSub-ADC+DAC\n->Sum->x2" if not last else f"Stage {k}\nSub-ADC only")
        if not last:
            edge(f"v{k}", (x + W, 3.0), (x + W + G, 3.0), "V", f"{r['vres']:.2f}V", (0, 0.1))
        edge(f"d{k}", (x + W / 2, 2.4), (x + W / 2, 1.5), "I", f"D{k}={r['d']}", (0.45, -0.1))
        box(f"r{k}", x + 0.1, 0.7, W - 0.2, 0.8, f"Shift Reg\n({n - k} CLK)")
        edge(f"r{k}", (x + W / 2, 0.7), (x + W / 2, 0.3), "I", head=False)
    edge("dec", (X0 + W / 2, 0.3), (xe + 0.3, 0.3), "I", f"sum={raw}", (0, 0.06))
    box("dec", xe + 0.3, -0.1, 1.9, 1.2, "DEC\n(shift & add)")
    edge("out", (xe + 2.2, 0.5), (xe + 2.7, 0.5), "I", f"Dout={code:0{n}b}", (0.45, 0.1))
    _legend(ax)
    ax.set_title(f"Signal flow: step {idx + 1} / {len(steps)}", fontsize=11)
    return fig


# ---------- ② 1段の中身（拡大） ----------
def draw_stage(k, r, sub):
    E = [  # id, 始点, 終点, 種類, ラベル, ラベル位置補正, 矢じり, 何ステップ目で流れるか
        ("in", (0.2, 2.0), (1.0, 2.0), "V", f"Vin={r['vin']:.3f}V", (0.2, 0.1), False, 0),
        ("br", (1.0, 2.0), (1.0, 3.7), "V", None, (0, 0), False, 0),
        ("sum_in", (1.0, 2.0), (5.4, 2.0), "V", "Vin (+)", (0, 0.1), True, 0),
        ("cmp", (1.0, 3.7), (1.6, 3.7), "V", None, (0, 0), True, 1),
        ("d_out", (2.7, 4.2), (2.7, 4.9), "I", f"D{k}={r['d']} -> Shift Reg", (1.3, -0.1), True, 1),
        ("d_dac", (3.8, 3.7), (5.0, 3.7), "I", f"D={r['d']}", (0, 0.1), True, 2),
        ("dac_out", (5.8, 3.2), (5.8, 2.4), "V", f"DAC={r['vdac']:.2f}V (-)", (1.0, -0.1), True, 2),
        ("sum_amp", (6.15, 2.0), (7.2, 2.0), "V", f"{r['vsub']:.3f}V", (0, 0.1), True, 3),
        ("amp_out", (8.4, 2.0), (9.7, 2.0), "V", f"Vres={r['vres']:.3f}V", (0, 0.1), True, 4),
    ]
    first = {e[0]: e[7] for e in E}
    act = {1: "sadc", 2: "dac", 3: "sum", 4: "amp"}.get(sub)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axis("off")
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0.8, 5.3)
    fc = lambda b: "#ffd54f" if act == b else "#e3eaf2"
    ax.add_patch(FancyBboxPatch((1.6, 3.2), 2.2, 1.0, boxstyle="round,pad=0.03", fc=fc("sadc"), ec="#345", lw=1.5))
    ax.text(2.7, 3.7, "Sub-ADC\n(2 comparators)", ha="center", va="center", fontsize=9)
    ax.add_patch(FancyBboxPatch((5.0, 3.2), 1.6, 1.0, boxstyle="round,pad=0.03", fc=fc("dac"), ec="#345", lw=1.5))
    ax.text(5.8, 3.7, "DAC", ha="center", va="center", fontsize=10)
    ax.add_patch(Circle((5.8, 2.0), 0.35, fc=fc("sum"), ec="#345", lw=1.5))
    ax.text(5.8, 2.0, "Σ", ha="center", va="center", fontsize=13)
    ax.add_patch(Polygon([(7.2, 1.3), (7.2, 2.7), (8.4, 2.0)], fc=fc("amp"), ec="#345", lw=1.5))
    ax.text(7.55, 2.0, "x2", ha="center", va="center", fontsize=11)
    for eid, p, q, kind, lab, lp, head, st_no in E:
        state = "future" if first[eid] > sub else ("active" if first[eid] == sub else "done")
        _arrow(ax, p, q, state, kind, lab, lp, head, fs=9)
    _legend(ax, "lower right")
    ax.set_title(f"Inside Stage {k}  (MDAC stage)", fontsize=11)
    return fig


# ---------- ③ スイッチトキャパシタMDAC ----------
def draw_sc(phase):
    p1, p2 = phase == 1, phase == 2
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.axis("off")
    ax.set_xlim(-0.3, 9.2)
    ax.set_ylim(0.9, 5.4)

    def wire(pts, active):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], c=V_COL if active else G_COL,
                lw=3.5 if active else 1.5, solid_capstyle="round")

    ax.text(-0.25, 3.2, "Vin", fontsize=11)
    ax.text(-0.25, 1.8, "Vdac", fontsize=11)
    wire([(0.4, 3.2), (1.0, 3.2)], p1)
    wire([(1.8, 3.2), (2.4, 3.2), (2.4, 2.5)], p1)
    wire([(0.4, 1.8), (1.0, 1.8)], p2)
    wire([(1.8, 1.8), (2.4, 1.8), (2.4, 2.5)], p2)
    for y, name, on in ((3.2, "S1 (phi1)", p1), (1.8, "S2 (phi2)", p2)):
        ax.add_patch(FancyBboxPatch((1.0, y - 0.35), 0.8, 0.7, boxstyle="round,pad=0.02",
                                    fc="#a5d6a7" if on else "#eceff1", ec="#345"))
        ax.text(1.4, y, name + ("\nON" if on else "\nOFF"), ha="center", va="center", fontsize=7)
    wire([(2.4, 2.5), (3.0, 2.5)], True)
    for x in (3.0, 3.3):
        ax.plot([x, x], [2.1, 2.9], c="#345", lw=3)
    wire([(3.3, 2.5), (4.2, 2.5), (4.2, 2.9), (4.6, 2.9)], True)
    ax.add_patch(Polygon([(4.6, 1.7), (4.6, 3.3), (6.6, 2.5)], fc="#e3eaf2", ec="#345", lw=1.5))
    ax.text(4.72, 2.83, "-", fontsize=12)
    ax.text(4.72, 2.05, "+", fontsize=12)
    ax.plot([4.6, 4.2], [2.1, 2.1], c="#345", lw=1.5)
    ax.text(3.75, 2.0, "GND", fontsize=8)
    wire([(4.2, 2.9), (4.2, 4.0)], p2)
    for y in (4.0, 4.25):
        ax.plot([3.9, 4.5], [y, y], c="#345", lw=3)
    wire([(4.2, 4.25), (4.2, 4.9), (7.4, 4.9), (7.4, 2.5)], p2)
    wire([(6.6, 2.5), (8.6, 2.5)], p2)
    ax.text(8.7, 2.45, "Vout", fontsize=11)
    ax.text(3.15, 3.05, "Cs", ha="center", fontsize=10)
    ax.text(4.75, 4.05, "Cf", fontsize=10)
    ax.text(3.0, 3.5, "X: virtual GND", fontsize=8)
    if p1:
        ax.text(5.0, 4.4, "Cf is reset (discharged)", fontsize=8)
    ax.set_title("Switched-capacitor MDAC  -  " + ("phi1: sample" if p1 else "phi2: amplify"), fontsize=11)
    return fig


# ---------- ④ 2相クロックとパイプライン ----------
def draw_timing(n, ns=3):
    names, cols = ["A", "B", "C"], ["#90caf9", "#a5d6a7", "#ffcc80"]
    T = n + 2 * (ns - 1) + 1
    rows_n = n + 2
    fig, ax = plt.subplots(figsize=(12, 0.55 * rows_n + 1.2))
    for h in range(T):
        for i, on in enumerate((h % 2 == 0, h % 2 == 1)):
            if on:
                ax.add_patch(plt.Rectangle((h, rows_n - 1 - i + 0.15), 1, 0.7, fc="#cfd8dc", ec="#345"))
    for k in range(1, n + 1):
        y = rows_n - 1 - (k + 1)
        for s in range(ns):
            for j, tag in ((k - 1 + 2 * s, "sample"), (k + 2 * s, "amp")):
                ax.add_patch(plt.Rectangle((j, y + 0.1), 1, 0.8, fc=cols[s], ec="#345",
                                           hatch="" if tag == "sample" else "//"))
                ax.text(j + 0.5, y + 0.5, f"{names[s]}\n{tag}", ha="center", va="center", fontsize=7)
    ax.set_xlim(0, T)
    ax.set_ylim(0, rows_n)
    ax.set_yticks([rows_n - 0.5 - i for i in range(rows_n)])
    ax.set_yticklabels(["phi1", "phi2"] + [f"Stage {k}" for k in range(1, n + 1)])
    ax.set_xticks(range(T + 1))
    ax.set_xlabel("half clock (phase)")
    ax.set_title("Two-phase operation: stage k amplifies while stage k+1 samples", fontsize=11)
    return fig


# ============================ 追加UI ============================
st.divider()
st.header("🔍 信号の流れをたどる（電圧の流れ・情報の流れ）")
st.caption("矢印を1本ずつたどりながら、**電圧（アナログ）＝青**と**情報（デジタル）＝赤**が"
           "どの順番で流れるかを確認できます。上のサイドバーの Vin / N がそのまま反映されます。")

_rw, _cd, _raw = run_adc(vin, n)
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🧭 全体の流れ（矢印）", "🔬 1段の中身", "⚡ MDACの電気の動き", "⏱ 2相クロックとパイプライン", "📝 キーワード別の解説"])

with tab1:
    steps = _build_steps(_rw, n, vin, _cd, _raw)
    i = st.slider("ステップ（矢印を1本ずつたどる）", 1, len(steps), 1, key="flow_step") - 1
    figf = draw_flow(n, _rw, vin, _cd, _raw, steps, i)
    st.pyplot(figf)
    plt.close(figf)
    icon = "🔵 電圧（アナログ）の流れ" if steps[i]["kind"] == "V" else "🔴 情報（デジタル）の流れ"
    st.info(f"**Step {i + 1}：{steps[i]['title']}**　（{icon}）\n\n{steps[i]['text']}")
    with st.expander("全ステップの一覧（電圧と情報の遷移順）"):
        for j, s in enumerate(steps):
            st.markdown(f"{'🔵' if s['kind'] == 'V' else '🔴'} **{j + 1}. {s['title']}**")
    st.caption("ポイント：電圧（青）は左から右へ1段ずつ受け渡され、"
               "情報（赤）は各段から下へ取り出されてシフトレジスタ → DEC → Dout へ集まります。")

with tab2:
    c1, c2 = st.columns(2)
    kk = c1.slider("拡大するステージ", 1, n - 1, 1, key="zoom_k")
    subs = ["① Vinが2方向へ分岐", "② Sub-ADC判定", "③ DACが電圧を生成", "④ Σで減算", "⑤ ×2アンプで残差増幅"]
    sname = c2.radio("流れの段階", subs, key="zoom_sub")
    si2 = subs.index(sname)
    r = _rw[kk - 1]
    figs = draw_stage(kk, r, si2)
    st.pyplot(figs)
    plt.close(figs)
    tx = [
        f"入力 **{r['vin']:.3f} V** が、①Sub-ADC へ向かう枝と ②Σ の＋入力へ向かう枝に**分岐**します（電圧は同じ）。",
        f"Sub-ADC の2個の比較器が 0.75 V / 1.25 V と比較 → **D{kk} = {r['d']}**。ここからデジタル情報（赤）が生まれます。",
        f"D{kk} = {r['d']} に応じて DAC が **{r['vdac']:.3f} V**（= D × Vref/4）を出力。デジタル→アナログへ戻る点です。",
        f"Σ で **{r['vin']:.3f} − {r['vdac']:.3f} = {r['vsub']:.3f} V**。粗い判定で表せた分を引き、細かい残りだけ（残差）にします。",
        f"アンプが×2 → **Vres = {r['vres']:.3f} V**。範囲が 0～Vref に戻るので、次段が同じ回路で処理できます。",
    ]
    st.info(tx[si2])
    st.caption("1つの段の中で、電圧は「分岐 → 引き算 → 2倍」と流れ、情報は「比較 → D → DAC／シフトレジスタ」と流れます。")

with tab3:
    ph = st.radio("クロック相", ["φ1：サンプル期間", "φ2：増幅期間"], horizontal=True, key="sc_phase")
    pn = 1 if ph.startswith("φ1") else 2
    figc = draw_sc(pn)
    st.pyplot(figc)
    plt.close(figc)
    r1 = _rw[0]
    cs, cf = 2.0, 1.0  # Cs/Cf = 2 → 利得2
    if pn == 1:
        st.info(f"**φ1**：S1がON。Cs の下側に Vin = {r1['vin']:.3f} V が加わり、Cs に電荷 "
                f"**Q = Cs × Vin = {cs * r1['vin']:.3f}**（Cf=1 に規格化）が蓄えられます。Cf は放電（リセット）。"
                "この時点で入力電圧が『電荷』として保持されます（＝S/H動作）。")
    else:
        st.info(f"**φ2**：S1がOFF、S2がON。Cs の下側が Vdac = {r1['vdac']:.3f} V に切り替わります。"
                f"Cs に残せる電荷は Cs × Vdac = {cs * r1['vdac']:.3f}。差の **{cs * (r1['vin'] - r1['vdac']):.3f}** が"
                f"オペアンプを通って Cf に移り、**Vout = {r1['vres']:.3f} V** になります。")
    st.latex(r"C_s V_{in} = C_s V_{dac} + C_f V_{out} \;\Rightarrow\; V_{out}=\frac{C_s}{C_f}\,(V_{in}-V_{dac}) = 2\,(V_{in}-V_{dac})")
    st.caption("電荷保存則がそのまま『引き算＋2倍』を実現します（Cs/Cf = 2、Vdac = D × Vref/4）。"
               "上の表示は Stage 1 の値です。")

with tab4:
    figt = draw_timing(n)
    st.pyplot(figt)
    plt.close(figt)
    st.markdown(
        "- 隣り合う段は**逆相**で動作します。Stage k が増幅（amp）して出力を出す**ちょうどその期間**に、"
        "Stage k+1 がその電圧をサンプル（sample）します。\n"
        "- そのため各段は「サンプル → 増幅」を繰り返しながら、**別々のサンプル（A,B,C）を同時に**処理できます。\n"
        "- 上図は実回路の2相動作の様子です。前の『パイプライン処理』の表は、各段にS/Hがある教科書的モデル"
        "（1段＝1クロック）で表しており、考え方は同じです。")

with tab5:
    _lines = " + ".join(f"{r['d']}×{weight(r['stage'], n)}" for r in _rw)
    with st.expander("1. 残差・MDAC ― 電圧の流れ", expanded=(focus == KW[1])):
        st.markdown(
            "**電圧の流れ**：Vin → 分岐 → ①Sub-ADC ②Σ(＋) ／ DAC電圧 → Σ(−) → 残差 → ×2 → Vres → 次段\n\n"
            "**情報の流れ**：Sub-ADC → D(0,1,2) → ①DAC（電圧を選ぶ）②シフトレジスタ（記録）\n\n"
            "**なぜ2倍？**：残差は 0～Vref/2 と狭くなっています。2倍で 0～Vref に戻せば、"
            "次段も全く同じ回路・同じしきい値で判定できます。")
    with st.expander("2. パイプライン処理 ― 各段が別サンプルを保持", expanded=(focus == KW[2])):
        st.markdown(
            "**電圧の流れ**：各段のアナログ出力（Vres）が次段のサンプル用キャパシタへ渡ります。"
            "各段には**別のサンプルの電圧**が保持されています。\n\n"
            "**情報の流れ**：サンプルAの D1, D2, … は1クロックずつ遅れて確定します。"
            "その間に後続のサンプルB, C も別の段で処理されています。\n\n"
            "**結果**：スループット＝毎クロック1サンプル、レイテンシ＝Nクロック。（『2相クロックとパイプライン』タブ参照）")
    with st.expander("3. 時差・誤差補正 (DEC) ― デジタル情報だけが流れる", expanded=(focus == KW[3])):
        st.markdown(
            "**情報の流れ**：D_k（2bit）は Stage k が確定した時刻に出て、**電圧は関与しません**。"
            "シフトレジスタ（Dフリップフロップ）を N−k クロック通り、加算器で合流します。\n\n"
            f"**現在の値での加算**：`{_lines} = {_raw}`（重みは2の冪。最下位2段は同じ重み）\n\n"
            "**なぜ誤差補正になる？**：段kの比較器が1段階誤って D_k が +1 大きくなると、残差は −1 V 小さくなり、"
            "次段の D_{k+1} が −2 になります。D_k の重みは D_{k+1} の2倍なので、加算すると "
            "**+1×2 − 2×1 = 0** で誤りが打ち消されます。ただし残差が 0～Vref を超えると打ち消せません。")
