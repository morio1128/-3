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
