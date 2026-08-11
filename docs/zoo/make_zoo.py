# Generates the Strict-Transversal Gate Zoo (docs/index.html) from zoo_data.json.
import json
import math
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "zoo_data.json").read_text())
BY = {d["name"]: d for d in DATA}

REPO = "https://github.com/Muuuun/qec-transversal"

# Human definitions per code (kept in sync with src/qec_transversal/codes.py).
DEFS = {
    "steane": "H<sub>X</sub> = H<sub>Z</sub> = parity checks of the [7,4,3] Hamming code. Self-dual, doubly even.",
    "c4-22": "H<sub>X</sub> = H<sub>Z</sub> = [1111]. The smallest error-detecting CSS code.",
    "c6-22": "H<sub>X</sub> = H<sub>Z</sub> = {111100, 110011}.",
    "iceberg-8": "H<sub>X</sub> = H<sub>Z</sub> = [11111111]: one global X and one global Z stabilizer.",
    "iceberg-12": "H<sub>X</sub> = H<sub>Z</sub> = all-ones on 12 qubits. Rate 5/6; the [[2m,2m−2,2]] family reaches rate → 1 at distance 2.",
    "rm256": "C<sub>X</sub> = C<sub>Z</sub> = RM(3,8), the middle Reed–Muller code on 256 points (check weights 32–256: powerful but decidedly not LDPC).",
    "cube-832": "[[8,3,2]] cube code: one global X stabilizer, Z faces of the cube. The all-T layer implements logical CCZ.",
    "qrm15": "X checks: the 4 coordinate-bit vectors of 1..15; Z checks add their 6 pairwise products. C<sub>X</sub> ⊂ C<sub>Z</sub> (triply even): the famous transversal-T code.",
    "tesseract": "C<sub>X</sub> = C<sub>Z</sub> = RM(1,4), the [16,5,8] first-order Reed–Muller code.",
    "rm64": "C<sub>X</sub> = C<sub>Z</sub> = RM(2,6), the middle Reed–Muller code on 64 points.",
    "grid-4x6": "Checks span {row<sub>i</sub> + col<sub>j</sub>} on a 4×6 cell grid; self-dual, doubly even.",
    "grid-6x8": "Checks span {row<sub>i</sub> + col<sub>j</sub>} on a 6×8 cell grid.",
    "toric-4": "Hypergraph product of the length-4 cyclic repetition code with itself.",
    "toric-10": "Hypergraph product of the length-10 cyclic repetition code with itself.",
    "surface-5": "Hypergraph product of the open 4×5 repetition chain with itself.",
    "bb72": "BB(ℓ=6, m=6): A = x³+y+y², B = y³+x+x².",
    "bb90": "BB(ℓ=15, m=3): A = x⁹+y+y², B = 1+x²+x⁷.",
    "bb108": "BB(ℓ=9, m=6): A = x³+y+y², B = y³+x+x².",
    "gross": "BB(ℓ=12, m=6): A = x³+y+y², B = y³+x+x². IBM's gross code.",
    "two-gross": "BB(ℓ=12, m=12): A = x³+y²+y⁷, B = y³+x+x². The two-gross code.",
    "bb360": "BB(ℓ=30, m=6): A = x⁹+y+y², B = y³+x²⁵+x²⁶.",
    "bb756": "BB(ℓ=21, m=18): A = x³+y¹⁰+y¹⁷, B = y⁵+x³+x¹⁹.",
    "bb54": "BB(ℓ=3, m=9): A = 1+y²+y⁴, B = y³+x+x².",
    "bb98-symmetric": "BB(ℓ=7, m=7): A = x+y³+y⁴, B = y+x³+x⁴. Symmetric: B(x,y) = A(y,x), so it carries the CZ/S-type fold-transversal gates of Eberhardt–Steffan.",
    "bb162-symmetric": "BB(ℓ=9, m=9): A = x³+y+y², B = y³+x+x². Symmetric.",
    "coprime30": "Coprime BB(ℓ=3, m=5), π = xy: a = 1+π+π², b = 1+π²+π⁷.",
    "coprime42": "Coprime BB(ℓ=3, m=7): a = 1+π²+π³, b = 1+π²+π¹⁰.",
    "coprime70": "Coprime BB(ℓ=5, m=7): a = 1+π+π⁵, b = 1+π+π¹².",
    "coprime126": "Coprime BB(ℓ=7, m=9): a = 1+π+π⁵⁸, b = 1+π¹³+π⁴¹.",
    "coprime154": "Coprime BB(ℓ=7, m=11): a = 1+π+π³¹, b = 1+π¹⁹+π⁵³.",
    "trivariate30": "Trivariate bicycle(ℓ=3, m=5), z = xy: A = x+z⁴, B = x+y²+z². Weight-5 checks.",
    "gb48": "GB(ℓ=24): a = 1+x²+x⁸+x¹⁵, b = 1+x²+x¹²+x¹⁷ (Panteleev–Kalachev A3).",
    "gb46": "GB(ℓ=23): a = 1+x⁵+x⁸+x¹², b = 1+x+x⁵+x⁷ (A4).",
    "gb126": "GB(ℓ=63): a = 1+x+x¹⁴+x¹⁶+x²², b = 1+x³+x¹³+x²⁰+x⁴² (A2).",
    "hgp-hamming": "Hypergraph product of two [7,4,3] Hamming codes.",
    "lacross65": "La-cross: HGP of the open seed 1+x+x³ (n=7) with itself.",
    "lacross400": "La-cross: HGP of the open seed 1+x+x⁴ (n=16) with itself.",
    "lifted-b1": "Lifted product over 𝔽₂[x]/(x⁶³−1): A[i,i]=x²⁷, A[i,i+5]=1, A[i,i+6]=x⁵⁴ (7×7 blocks), B=(1+x+x⁶)·I₇ (Panteleev–Kalachev B1).",
    "kasai-binary-294": "Komoto–Kasai girth-12 orthogonal QC pair, J=2, L=6, P=49 (shifts fℓ=2^ℓ, gℓ=2^(ℓ+3)).",
    "kasai-binary-1104": "Komoto–Kasai girth-12 orthogonal QC pair, J=2, L=8, P=138.",
    "kasai-gf256-2352": "The L=6, P=49 pair lifted to GF(256) labels (canonical separable assignment) and expanded through 8×8 companion matrices.",
}

FAMILY_GROUPS = [
    ("Bivariate bicycle codes", "Bravyi et al., Nature 627, 778 (2024), arXiv:2308.07915; symmetric instances from Eberhardt–Steffan, arXiv:2407.03973; [[54,8,6]] from arXiv:2408.10001",
     ["bb72", "bb90", "bb108", "gross", "two-gross", "bb360", "bb756", "bb54", "bb98-symmetric", "bb162-symmetric"]),
    ("Coprime & trivariate bicycle codes", "Wang–Mueller, arXiv:2408.10001; multivariate bicycle, arXiv:2406.19151",
     ["coprime30", "coprime42", "coprime70", "coprime126", "coprime154", "trivariate30"]),
    ("Generalized bicycle codes", "Panteleev–Kalachev, Quantum 5, 585 (2021), arXiv:1904.02703, App. B",
     ["gb48", "gb46", "gb126"]),
    ("Hypergraph, lifted products & La-cross", "Tillich–Zémor arXiv:0903.0566; Panteleev–Kalachev arXiv:1904.02703; Pecorari et al., Nat. Commun. 16, 1111 (2025), arXiv:2404.13010",
     ["hgp-hamming", "lifted-b1", "lacross65", "lacross400"]),
    ("Kasai quasi-cyclic codes", "Komoto–Kasai, npj Quantum Inf. 11, 154 (2025), arXiv:2412.21171; girth-12 pair from arXiv:2501.13444",
     ["kasai-binary-294", "kasai-binary-1104", "kasai-gf256-2352"]),
    ("Topological controls", "Toric and surface codes as hypergraph products of repetition codes",
     ["toric-4", "toric-10", "surface-5"]),
]

POSITIVE = ["steane", "c4-22", "c6-22", "cube-832", "iceberg-8", "iceberg-12", "qrm15",
            "tesseract", "rm64", "rm256", "grid-4x6", "grid-6x8"]
NEGATIVE = [nm for _, _, names in FAMILY_GROUPS for nm in names]

# Literature context for the fold layer (citations shown alongside the
# machine-certified verdicts computed from zoo_data's 'fold' records).
FOLD_LIT = {
    "bb72": "symmetric BB: H-type fold + CZ/S-type fold gates (arXiv:2407.03973)",
    "bb90": "H-type fold gate via the BB ZX-duality; Swap automorphisms (arXiv:2407.03973)",
    "bb108": "H-type fold gate via the BB ZX-duality; Swap automorphisms (arXiv:2407.03973)",
    "gross": "fixed-point-free duality: CZ-matching fold layer (arXiv:2407.03973, 2608.05688)",
    "two-gross": "H-type fold gate via the BB ZX-duality (arXiv:2407.03973)",
    "bb360": "H-type fold gate via the BB ZX-duality (arXiv:2407.03973)",
    "bb756": "H-type fold gate via the BB ZX-duality (arXiv:2407.03973)",
    "bb54": "H-type fold gate via the BB ZX-duality (arXiv:2407.03973)",
    "bb98-symmetric": "symmetric BB: fold gate group C2 x Sp2(F8) (arXiv:2407.03973)",
    "bb162-symmetric": "symmetric BB: fold gate group Sp2(F4) x (Sp2(F4):C2) (arXiv:2407.03973)",
    "coprime30": "two-block ZX-duality fold gates (arXiv:2202.06647, 2407.03973)",
    "coprime42": "two-block ZX-duality fold gates (arXiv:2202.06647, 2407.03973)",
    "coprime70": "two-block ZX-duality fold gates (arXiv:2202.06647, 2407.03973)",
    "coprime126": "two-block ZX-duality fold gates (arXiv:2202.06647, 2407.03973)",
    "coprime154": "two-block ZX-duality fold gates (arXiv:2202.06647, 2407.03973)",
    "trivariate30": "two-block ZX-duality fold gates (arXiv:2202.06647)",
    "gb48": "two-block ZX-duality fold gates (arXiv:2202.06647)",
    "gb46": "two-block ZX-duality fold gates (arXiv:2202.06647)",
    "gb126": "two-block ZX-duality fold gates (arXiv:2202.06647)",
    "hgp-hamming": "symmetric-HGP swap duality fold gates (arXiv:2204.10812)",
    "lacross65": "symmetric-HGP swap duality fold gates (arXiv:2204.10812)",
    "lacross400": "symmetric-HGP swap duality fold gates (arXiv:2204.10812)",
    "lifted-b1": "two-block transpose duality; fold construction applies (arXiv:2202.06647)",
    "toric-4": "folded-code H and S gates (Moussa 2016; arXiv:2202.06647)",
    "toric-10": "folded-code H and S gates (Moussa 2016; arXiv:2202.06647)",
    "surface-5": "folded surface-code S gate (Moussa 2016)",
}



def nkd(d):
    dd = "?" if d["d"] is None else (("≤" + str(d["d"])) if d["d_ub"] else str(d["d"]))
    return f"[[{d['n']},{d['k']},{dd}]]"


def bits(support, n, wrap=40):
    cells = []
    filled = set(support)
    for i in range(n):
        cls = "b1" if i in filled else "b0"
        cells.append(f'<i class="{cls}"></i>')
        if (i + 1) % wrap == 0 and i + 1 < n:
            cells.append("<br>")
    return (f'<span class="bits" role="img" aria-label="binary vector, weight '
            f'{len(support)} of {n}">{"".join(cells)}</span>')


def matrix2(m):
    rows = ["&thinsp;".join(str(v) for v in row) for row in m]
    return '<span class="mat">(' + "&nbsp;;&nbsp;".join(rows) + ")</span>"


def strict_positive(d):
    st = d["structure"]
    return st["logically_nontrivial_rank_A_Z"] + st["logically_nontrivial_rank_A_X"] > 0


def fold_state(d):
    """'strict' | 'fold' (machine-certified) | 'none'."""
    if strict_positive(d):
        return "strict"
    f = d.get("fold")
    if f and f["certified_dualities"] > 0 and f["nontrivial_fold_generators"] > 0:
        return "fold"
    return "none"


def fold_order_text(d):
    f = d.get("fold")
    if not f:
        return ""
    cg = f["combined_group"]
    if cg["exact"]:
        return str(cg["order"])
    if cg["lower_bound"]:
        return f"&ge;{cg['lower_bound']}"
    return "?"


def diag_level(d):
    h = d.get("hierarchy")
    if not h:
        return 0
    return max(h["Z"]["max_level"] or 0, h["X"]["max_level"] or 0)


def has_t(d):
    h = d.get("hierarchy")
    return bool(h and (h["Z"]["has_t_level_gate"] or h["X"]["has_t_level_gate"]))


def aut_info(d):
    return d.get("automorphisms")


def merit(d):
    """kd^2/n, or None when the distance is unknown."""
    if d["d"] is None:
        return None
    return d["k"] * d["d"] ** 2 / d["n"]


# ---------------------------------------------------------------- entries ---

def trivial_entry(name):
    d = BY[name]
    assert d["dim_AZ"] == 0 and d["dim_AX"] == 0
    assert d["rank_MZ"] == d["n"] and d["rank_MX"] == d["n"]
    f = d.get("fold")
    if f and fold_state(d) == "fold":
        labels = "; ".join(
            f'{escape(a["label"])} ({a["pairs"]} pairs, {a["fixed_points"]} fixed, '
            f'dim S<sub>M</sub><sup>Z</sup>={a["dim_S_MZ"]}, '
            f'dim S<sub>M</sub><sup>X</sup>={a["dim_S_MX"]}'
            + (", fold-H nontrivial" if a["fold_hadamard_nontrivial"] else "")
            + ")"
            for a in f["analyses"] if a["is_zx_duality"]
        )
        fold_line = (
            f'<p class="cert"><b>Fold.</b> '
            f'{f["certified_dualities"]} certified ZX-dualit{"y" if f["certified_dualities"]==1 else "ies"}: '
            f'{labels}. Combined logical group order {fold_order_text(d)}.</p>'
        )
    elif f and f["certified_dualities"] > 0:
        fold_line = (
            f'<p class="cert"><b>Fold.</b> {f["certified_dualities"]} ZX-dualit'
            f'{"y" if f["certified_dualities"]==1 else "ies"} certified, but every legal fold '
            f'layer acts as the logical identity.</p>'
        )
    elif f:
        fold_line = (
            f'<p class="cert"><b>Fold.</b> {f["candidates_tested"]} structural duality '
            f'candidate(s) tested — none certified; no fold gates known from any source.</p>'
        )
    else:
        fold_line = ""
    return f"""
<details class="entry" id="{name}">
  <summary>
    <span class="ename">{escape(name)}</span>
    <span class="nkd">{nkd(d)}</span>
    <span class="chip none">no strict gates</span>
  </summary>
  <div class="ebody">
    <p class="def">{DEFS[name]}</p>
    <p class="cert"><b>Strict.</b>
      rank&nbsp;M<sub>Z</sub> = {d['rank_MZ']} = n and rank&nbsp;M<sub>X</sub> = {d['rank_MX']} = n,
      so A<sub>Z</sub> = ker&nbsp;M<sub>Z</sub> = {{0}} and A<sub>X</sub> = ker&nbsp;M<sub>X</sub> = {{0}}.
      By the <a href="#method">completeness theorem</a> the strict-transversal group is exactly
      the Pauli group. <span class="t">verified in {d['seconds']:.2f}&thinsp;s</span></p>
    {fold_line}
  </div>
</details>"""


BEST_RATE = max(BY[nm]["k"] / BY[nm]["n"] for nm in POSITIVE)
BEST_EFF = max(merit(BY[nm]) for nm in POSITIVE)


def positive_entry(name, extra=""):
    d = BY[name]
    n = d["n"]
    leader = ""
    if abs(d["k"] / d["n"] - BEST_RATE) < 1e-9:
        g = math.gcd(d["k"], d["n"])
        leader += (f' <span class="chip rate" title="highest encoding rate among codes with '
                   f'strict-transversal gates">best rate {d["k"]//g}/{d["n"]//g}</span>')
    if merit(d) is not None and abs(merit(d) - BEST_EFF) < 1e-9:
        leader += (f' <span class="chip rate" title="highest kd²/n among codes with '
                   f'strict-transversal gates">best kd²/n = {merit(d):.0f}</span>')
    gens_html = []
    for g in d["generators"]:
        gate = "√Z" if g["family"] == "Z" else "√X"
        label = "logical identity" if g["logical_identity"] else "acts on logicals"
        logical = ""
        if g["logical"] is not None and not g["logical_identity"]:
            logical = f' &nbsp;λ = {matrix2(g["logical"])}'
        gens_html.append(
            f'<div class="gen"><span class="glabel">{gate} on</span> {bits(g["support"], n)}'
            f' <span class="gmeta">weight {g["weight"]} — {label}{logical}</span></div>'
        )
    order = d["order"]
    star = ' <span class="star" title="full logical Clifford group">★</span>' if d["is_full"] else ""
    full = " = <b>full logical Clifford group</b> Sp(2k,2) mod Paulis" if d["is_full"] else ""
    return f"""
<article class="entry has scard" id="{name}">
  <header>
    <span class="ename">{escape(name)}{star}</span>
    <span class="nkd">{nkd(d)}</span>{leader}
    <span class="chip yes">gates exist</span>
  </header>
  <p class="def">{DEFS[name]}</p>
  <div class="gens">{''.join(gens_html)}</div>
  <p class="cert"><b>Strict.</b>
    dim&nbsp;A<sub>Z</sub> = {d['dim_AZ']} (rank&nbsp;M<sub>Z</sub> = {d['rank_MZ']} of n = {n}),
    dim&nbsp;A<sub>X</sub> = {d['dim_AX']}.
    Logical group order {order}{full}.{extra}</p>
</article>"""


# ----------------------------------------------------------------- census ---

def census_row(name, has):
    d = BY[name]
    chip = ('<span class="chip yes">yes</span>' if has else
            '<span class="chip none">none</span>')
    star = ' <span class="star" title="full logical Clifford group">★</span>' if d["is_full"] else ""
    dsort = 0 if d["d"] is None else d["d"]
    rate = d["k"] / d["n"]
    lvl = diag_level(d)
    lvl_cell = {3: '<span class="chip yes">T (3)</span>', 2: "S (2)", 1: "Pauli", 0: "—"}[lvl]
    aut = aut_info(d)
    aut_cell = "—" if not aut else str(aut["qubit_group_order"])
    aut_sort = 0 if not aut else aut["qubit_group_order"]
    state = fold_state(d)
    f = d.get("fold")
    if f and f["certified_dualities"] and f["nontrivial_fold_generators"]:
        fold_cell = f'<span class="chip yes">order {fold_order_text(d)}</span>'
        fold_sort = 2 if state == "strict" else 1
    elif f:
        fold_cell, fold_sort = '<span class="chip none">none</span>', 0
    else:
        fold_cell, fold_sort = "—", 0
    eff = merit(d)
    eff_cell = "—" if eff is None else (("≤" if d["d_ub"] else "") + f"{eff:.1f}")
    return (f'<tr data-name="{name}" data-family="{escape(d["family"])}" data-n="{d["n"]}" '
            f'data-k="{d["k"]}" data-d="{dsort}" data-az="{d["dim_AZ"]}" data-ax="{d["dim_AX"]}" '
            f'data-order="{d["order"]}" data-rate="{rate:.4f}" data-eff="{0 if eff is None else round(eff, 2)}" '
            f'data-fold="{fold_sort}" data-lvl="{lvl}" data-aut="{aut_sort}" data-gates="{"yes" if has else "no"}">'
            f'<td><a href="#{name}">{escape(name)}</a>{star}</td>'
            f'<td class="mono">{nkd(d)}</td><td>{escape(d["family"])}</td>'
            f'<td class="num">{rate:.3f}</td>'
            f'<td class="num">{eff_cell}</td>'
            f'<td class="num">{d["dim_AZ"]} / {d["dim_AX"]}</td>'
            f'<td class="num">{d["order"]}</td><td>{chip}</td><td>{fold_cell}</td>'
            f'<td>{lvl_cell}</td><td class="num">{aut_cell}</td></tr>')


# ------------------------------------------------------------------ chart ---

def scatter_svg():
    W, H = 720, 380
    L, R, T, B = 52, 16, 14, 40
    xmin, xmax = math.log10(3.5), math.log10(3000)
    ymin, ymax = math.log10(0.9), math.log10(1100)

    def X(n):
        return L + (math.log10(n) - xmin) / (xmax - xmin) * (W - L - R)

    def Y(k):
        return H - B - (math.log10(k) - ymin) / (ymax - ymin) * (H - T - B)

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="Scatter chart of all {len(DATA)} codes: physical qubits n against logical qubits k">']
    for n in [10, 100, 1000]:
        x = X(n)
        parts.append(f'<line class="grid" x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H-B}"/>')
        parts.append(f'<text class="tick" x="{x:.1f}" y="{H-B+18}" text-anchor="middle">{n}</text>')
    for k in [1, 10, 100, 1000]:
        y = Y(k)
        parts.append(f'<line class="grid" x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}"/>')
        parts.append(f'<text class="tick" x="{L-8}" y="{y+4:.1f}" text-anchor="end">{k}</text>')
    parts.append(f'<text class="axis" x="{(L+W-R)/2:.0f}" y="{H-6}" text-anchor="middle">physical qubits n</text>')
    parts.append(f'<text class="axis" x="14" y="{(T+H-B)/2:.0f}" text-anchor="middle" '
                 f'transform="rotate(-90 14 {(T+H-B)/2:.0f})">logical qubits k</text>')
    x1, y1 = X(4), Y(2)
    x2, y2 = X(2000), Y(1000)
    parts.append(f'<line class="guide" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')
    parts.append(f'<text class="guidelabel" x="{X(300):.1f}" y="{Y(150)-8:.1f}">k = n/2</text>')
    for d in DATA:
        name = d["name"]
        state = fold_state(d)
        if state == "strict":
            cls, r = "pt-strict", 7
            title = f"{name} {nkd(d)} — strict gates"
        elif state == "fold":
            f = d["fold"]
            cls, r = "pt-fold", 6
            title = (f"{name} {nkd(d)} — fold gates only: "
                     f"{f['certified_dualities']} certified duality(ies), gate group order "
                     f"{fold_order_text(d)}")
        else:
            f = d.get("fold")
            tested = f["candidates_tested"] if f else 0
            cls, r = "pt-none", 5
            title = (f"{name} {nkd(d)} — no gates in any class "
                     f"({tested} duality candidates tested; no duality exists)")
        aut = aut_info(d)
        if aut:
            title += (f" | automorphism group order {aut['qubit_group_order']}"
                      + ("" if aut['duality_exists'] else "; no ZX-duality"))
        if has_t(d):
            title += " | T-LEVEL GATE (transversal T/CCZ family)"
        x, y = X(d["n"]), Y(max(d["k"], 1))
        ring = (f'<circle class="pt-t" cx="{x:.1f}" cy="{y:.1f}" r="{r + 3.5}"/>'
                if has_t(d) else "")
        parts.append(
            f'<a href="#{name}">{ring}<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r}">'
            f'<title>{escape(title)}</title></circle></a>'
        )
    parts.append("</svg>")
    return "".join(parts)


# ------------------------------------------------------------------- page ---

groups_html = []
for title, src, names in FAMILY_GROUPS:
    entries = "".join(trivial_entry(nm) for nm in names)
    count = len(names)
    groups_html.append(f"""
<section class="famgroup">
  <h3>{escape(title)} <span class="fmeta">{count} codes · all trivial</span></h3>
  <p class="src">{src}</p>
  {entries}
</section>""")

qrm_extra = (" Four of the five √Z layers act as the logical identity (they lie on stabilizer-"
             "supported patterns); the fifth is the logical phase gate S̄ inherited from the "
             "code's transversal T.")
positives_html = "".join(
    positive_entry(nm, qrm_extra if nm == "qrm15" else "") for nm in POSITIVE
)

census_rows = ("".join(census_row(nm, False) for nm in NEGATIVE)
               + "".join(census_row(nm, True) for nm in POSITIVE))
assert {nm for nm in POSITIVE} == {d["name"] for d in DATA if strict_positive(d)}, \
    "POSITIVE display list out of sync with certified data"

largest = max(DATA, key=lambda d: d["n"])
FAMILY_COUNT = len({d["family"] for d in DATA})
FOLD_CERTIFIED = sum(1 for d in DATA if fold_state(d) == "fold")
T_CERTIFIED = sum(1 for d in DATA if has_t(d))
BEST_EFF_CODE = max((BY[nm] for nm in POSITIVE), key=merit)
BEST_RATE_CODE = max((BY[nm] for nm in POSITIVE), key=lambda d: d["k"] / d["n"])

CSS = """
:root {
  --paper:#FFFFFF; --ink:#111111; --muted:#666666; --rule:#E2E2E2;
  --accent:#111111; --accent-ink:#111111; --chipno-bg:#F1F1F1; --chipno-tx:#666666;
  --chipyes-bg:#111111; --chipyes-tx:#FFFFFF; --entry:#FFFFFF; --code-bg:#F5F5F5;
  --bit0:#E6E6E6; --bit1:#111111;
  --nav-bg:rgba(255,255,255,.93); --shadow:0 1px 3px rgba(0,0,0,.07);
  --c-strict:#1F7A4D; --c-fold:#2B6CB0; --c-t:#B8860B;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper:#111111; --ink:#EDEDED; --muted:#9A9A9A; --rule:#2C2C2C;
    --accent:#EDEDED; --accent-ink:#EDEDED; --chipno-bg:#1D1D1D; --chipno-tx:#9A9A9A;
    --chipyes-bg:#EDEDED; --chipyes-tx:#111111; --entry:#181818; --code-bg:#1D1D1D;
    --bit0:#2C2C2C; --bit1:#EDEDED;
    --nav-bg:rgba(17,17,17,.93); --shadow:0 1px 3px rgba(0,0,0,.5);
    --c-strict:#4CC38A; --c-fold:#6CA9E8; --c-t:#E8C158;
  }
}
:root[data-theme="dark"] {
  --paper:#111111; --ink:#EDEDED; --muted:#9A9A9A; --rule:#2C2C2C;
  --accent:#EDEDED; --accent-ink:#EDEDED; --chipno-bg:#1D1D1D; --chipno-tx:#9A9A9A;
  --chipyes-bg:#EDEDED; --chipyes-tx:#111111; --entry:#181818; --code-bg:#1D1D1D;
  --bit0:#2C2C2C; --bit1:#EDEDED;
  --nav-bg:rgba(17,17,17,.93); --shadow:0 1px 3px rgba(0,0,0,.5);
  --c-strict:#4CC38A; --c-fold:#6CA9E8; --c-t:#E8C158;
}
* { box-sizing:border-box; }
html { scroll-behavior:smooth; scroll-padding-top:64px; }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior:auto; } }
body {
  background:var(--paper); color:var(--ink); margin:0;
  font:16px/1.6 "Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
}
.mono, code, .mat, .nkd, .cert, table, .bits, .glabel, .gmeta, .stat b, .filter input {
  font-family: ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
nav.top {
  position:sticky; top:0; z-index:10; background:var(--nav-bg);
  backdrop-filter:blur(8px); border-bottom:1px solid var(--rule);
}
nav.top .inner {
  max-width:1000px; margin:0 auto; padding:.55rem 1.25rem;
  display:flex; align-items:center; gap:1.2rem; flex-wrap:wrap;
}
nav.top .brand { font-weight:700; font-size:.95rem; color:var(--ink); text-decoration:none; }
nav.top .brand span { color:var(--accent); }
nav.top a.nl {
  font-family:ui-monospace,Menlo,monospace; font-size:.74rem; letter-spacing:.05em;
  color:var(--muted); text-decoration:none; text-transform:uppercase;
}
nav.top a.nl:hover, nav.top a.nl:focus-visible { color:var(--accent-ink); }
main { max-width:1000px; margin:0 auto; padding:2.6rem 1.25rem 5rem; }
.narrow { max-width:76ch; }
.eyebrow { font-family:ui-monospace,Menlo,monospace; font-size:.72rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted); }
h1 { font-size:clamp(2rem,5vw,2.9rem); line-height:1.1; margin:.35rem 0 .7rem; font-weight:600; text-wrap:balance; }
.standfirst { font-size:1.12rem; color:var(--muted); margin:0 0 1.4rem; max-width:64ch; }
.stats { display:flex; gap:.8rem; flex-wrap:wrap; margin:0 0 1rem; }
.stat {
  background:var(--entry); border:1px solid var(--rule); border-radius:4px;
  padding:.55rem 1rem .6rem; box-shadow:var(--shadow); min-width:8.5rem;
}
.stat b { display:block; font-size:1.45rem; color:var(--ink); font-variant-numeric:tabular-nums; }
.stat span { font-size:.72rem; color:var(--muted); letter-spacing:.05em; text-transform:uppercase; }
.stat span a { color:var(--ink); font-family:ui-monospace,Menlo,monospace; text-transform:none; letter-spacing:0; }
h2 { font-size:1.45rem; margin:3rem 0 .6rem; text-wrap:balance; }
h2 .no { color:var(--accent); font-family:ui-monospace,Menlo,monospace; font-size:.95rem;
  vertical-align:.18em; margin-right:.45em; }
h3 { font-size:1.1rem; margin:1.8rem 0 .1rem; }
.fmeta { font-family:ui-monospace,Menlo,monospace; font-size:.72rem; color:var(--muted);
  font-weight:400; margin-left:.5em; }
p { margin:.6rem 0; }
.math { background:var(--code-bg); border-left:3px solid var(--accent); padding:.8rem 1.1rem;
  margin:1rem 0; font-size:.96rem; overflow-x:auto; }
.math p { margin:.4rem 0; }
.src { font-size:.78rem; font-family:ui-monospace,Menlo,monospace; color:var(--muted); margin:.15rem 0 .7rem; }
.chartcard { background:var(--entry); border:1px solid var(--rule); border-radius:4px;
  padding:1rem; box-shadow:var(--shadow); overflow-x:auto; }
.chartcard svg { width:100%; height:auto; min-width:560px; display:block; }
.grid { stroke:var(--rule); stroke-width:1; }
.tick, .axis { fill:var(--muted); font-family:ui-monospace,Menlo,monospace; font-size:11px; }
.pt-none { fill:var(--paper); stroke:var(--muted); stroke-width:1.2; }
.pt-fold { fill:var(--c-fold); stroke:var(--c-fold); stroke-width:1.2; opacity:.92; }
.pt-strict { fill:var(--c-strict); stroke:var(--c-strict); stroke-width:1.2; }
.pt-t { fill:none; stroke:var(--c-t); stroke-width:2.2; }
.ring { display:inline-block; width:12px; height:12px; border-radius:50%; border:2.2px solid var(--c-t); margin-right:.35em; vertical-align:-2px; }
.guide { stroke:var(--muted); stroke-width:1; stroke-dasharray:5 4; }
.guidelabel { fill:var(--muted); font-family:ui-monospace,Menlo,monospace; font-size:11px; }
svg a:hover circle, svg a:focus circle { stroke-width:3; }
.legend { display:flex; gap:1.4rem; font-size:.78rem; color:var(--muted);
  font-family:ui-monospace,Menlo,monospace; margin:.6rem 0 0; flex-wrap:wrap; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:.35em; }
.dot.strict { background:var(--c-strict); }
.dot.fold { background:var(--c-fold); }
.dot.none { background:var(--paper); border:1.2px solid var(--muted); }
.filter { margin:.8rem 0 .4rem; }
.filter input {
  width:100%; max-width:34rem; font-size:.86rem; padding:.5rem .8rem;
  border:1px solid var(--rule); border-radius:4px; background:var(--entry); color:var(--ink);
}
.filter input:focus-visible { outline:2px solid var(--accent); outline-offset:1px; }
.hint { font-size:.74rem; color:var(--muted); font-family:ui-monospace,Menlo,monospace; margin:.3rem 0 0; }
.count { font-size:.78rem; color:var(--muted); font-family:ui-monospace,Menlo,monospace; }
.tablewrap { overflow-x:auto; margin:.6rem 0 1rem; background:var(--entry);
  border:1px solid var(--rule); border-radius:4px; box-shadow:var(--shadow); }
table { border-collapse:collapse; font-size:.82rem; width:100%; min-width:680px; }
th { text-align:left; font-weight:600; letter-spacing:.06em; text-transform:uppercase;
  font-size:.66rem; color:var(--muted); border-bottom:2px solid var(--rule);
  padding:.55rem .8rem; cursor:pointer; user-select:none; white-space:nowrap; }
th:hover { color:var(--accent-ink); }
.defs th { cursor:default; }
.defs td { vertical-align:top; font-size:.78rem; }
.defs tr.ours td { border-top:2px solid var(--ink); border-bottom:2px solid var(--ink); }
th .arr { opacity:.6; }
td { border-bottom:1px solid var(--rule); padding:.42rem .8rem; }
tr:last-child td { border-bottom:none; }
td.num { font-variant-numeric:tabular-nums; }
td a { color:var(--ink); text-decoration:underline; text-underline-offset:2px; font-weight:600; }
td a:hover, td a:focus-visible { text-decoration:underline; }
.chip { display:inline-block; font-family:ui-monospace,Menlo,monospace; font-size:.66rem;
  padding:.12rem .55rem; border-radius:10px; white-space:nowrap; }
.chip.none { background:var(--chipno-bg); color:var(--chipno-tx); }
.chip.yes  { background:var(--chipyes-bg); color:var(--chipyes-tx); font-weight:700; }
.chip.rate { background:var(--paper); color:var(--ink); border:1.2px solid var(--ink); font-weight:700; }
.star { color:var(--accent); }
.entry { background:var(--entry); border:1px solid var(--rule); border-radius:4px;
  margin:.55rem 0; box-shadow:var(--shadow); }
details.entry summary {
  display:flex; align-items:baseline; gap:.8rem; flex-wrap:wrap; cursor:pointer;
  padding:.6rem 1rem; list-style:none;
}
details.entry summary::-webkit-details-marker { display:none; }
details.entry summary::before {
  content:"+"; font-family:ui-monospace,Menlo,monospace; color:var(--accent);
  font-weight:700; width:1em;
}
details.entry[open] summary::before { content:"−"; }
details.entry summary:focus-visible { outline:2px solid var(--accent); outline-offset:-2px; }
.ename { font-weight:700; font-size:.95rem; }
.nkd { color:var(--muted); font-size:.82rem; }
summary .chip, .scard header .chip { margin-left:auto; }
.ebody { padding:0 1rem .8rem 2.1rem; }
.scard { padding:.8rem 1.1rem; border-left:4px solid var(--accent); }
.scard header { display:flex; align-items:baseline; gap:.8rem; flex-wrap:wrap; }
.def { font-size:.93rem; margin:.45rem 0; }
.cert { font-size:.78rem; color:var(--muted); margin:.45rem 0 0; }
.cert b { color:var(--ink); }
.cert a { color:var(--ink); text-decoration-thickness:1px; text-underline-offset:2px; }
.t { float:right; opacity:.8; }
.gens { display:flex; flex-direction:column; gap:.45rem; margin:.6rem 0; }
.gen { font-size:.78rem; overflow-x:auto; }
.glabel { color:var(--accent-ink); font-weight:700; margin-right:.4em; }
.gmeta { color:var(--muted); margin-left:.5em; }
.bits { display:inline-block; line-height:14px; vertical-align:middle; }
.bits i { display:inline-block; width:10px; height:10px; margin:1px; border-radius:1px; }
.bits .b0 { background:var(--bit0); }
.bits .b1 { background:var(--bit1); }
.mat { white-space:nowrap; }
.callout { border:1px solid var(--rule); border-left:4px solid var(--accent); border-radius:4px;
  padding:.9rem 1.1rem; margin:1.2rem 0; background:var(--entry); font-size:.95rem;
  box-shadow:var(--shadow); }
.callout .eyebrow { display:block; margin-bottom:.3rem; }
pre { background:var(--code-bg); padding:.8rem 1rem; overflow-x:auto; font-size:.82rem;
  border-radius:4px; }
footer { margin-top:3.5rem; border-top:1px solid var(--rule); padding-top:1rem;
  font-size:.8rem; color:var(--muted); }
footer ul { padding-left:1.2rem; margin:.4rem 0; }
a { color:var(--ink); text-decoration-thickness:1px; text-underline-offset:2px; }
"""

JS = """
(function () {
  var input = document.getElementById('q');
  var table = document.getElementById('census');
  var rows = Array.prototype.slice.call(table.tBodies[0].rows);
  var count = document.getElementById('rowcount');
  var NUM = { n: 'n', k: 'k', d: 'd', az: 'az', ax: 'ax', order: 'order', rate: 'rate', eff: 'eff', lvl: 'lvl', aut: 'aut' };

  function applyFilter() {
    var tokens = input.value.toLowerCase().split(/\\s+/).filter(Boolean);
    var shown = 0;
    rows.forEach(function (tr) {
      var ok = tokens.every(function (tok) {
        var m = tok.match(/^(n|k|d|az|ax|order|rate|eff|lvl|aut)(>=|<=|>|<|=)(\\d+(?:\\.\\d+)?)$/);
        if (m) {
          var v = parseFloat(tr.dataset[NUM[m[1]]]);
          var t = parseFloat(m[3]);
          if (m[2] === '>=') return v >= t;
          if (m[2] === '<=') return v <= t;
          if (m[2] === '>') return v > t;
          if (m[2] === '<') return v < t;
          return v === t;
        }
        var g = tok.match(/^gates:(yes|no)$/);
        if (g) return tr.dataset.gates === g[1];
        return (tr.dataset.name + ' ' + tr.dataset.family).toLowerCase().indexOf(tok) !== -1;
      });
      tr.style.display = ok ? '' : 'none';
      if (ok) shown++;
    });
    count.textContent = shown + ' of ' + rows.length + ' codes';
  }
  input.addEventListener('input', applyFilter);
  applyFilter();

  var dir = {};
  Array.prototype.forEach.call(table.tHead.rows[0].cells, function (th) {
    var key = th.dataset.sort;
    if (!key) return;
    th.addEventListener('click', function () {
      dir[key] = -(dir[key] || -1);
      var numeric = key !== 'name' && key !== 'family' && key !== 'gates';
      rows.sort(function (a, b) {
        var x = a.dataset[key], y = b.dataset[key];
        if (numeric) { x = parseFloat(x); y = parseFloat(y); }
        return (x > y ? 1 : x < y ? -1 : 0) * dir[key];
      });
      rows.forEach(function (r) { table.tBodies[0].appendChild(r); });
      Array.prototype.forEach.call(table.tHead.rows[0].cells, function (c) {
        var a = c.querySelector('.arr'); if (a) a.textContent = '';
      });
      var arr = th.querySelector('.arr');
      if (arr) arr.textContent = dir[key] === 1 ? ' ↑' : ' ↓';
    });
  });

  // Open a collapsed entry when it is the navigation target.
  function openTarget() {
    var el = location.hash && document.querySelector(location.hash);
    if (el && el.tagName === 'DETAILS') el.open = true;
  }
  window.addEventListener('hashchange', openTarget);
  openTarget();
})();
"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strict-Transversal Gate Zoo</title>
<meta name="description" content="A certified census of strict-transversal Clifford gates
across 41 well-known quantum LDPC and CSS codes: exact nonexistence certificates and exact
gate solutions, computed with qec-transversal.">
<style>{CSS}</style>
</head>
<body>
<nav class="top"><div class="inner">
  <a class="brand" href="#top">Transversal Gate <span>Zoo</span></a>
  <a class="nl" href="#definitions">Definitions</a>
  <a class="nl" href="#chart">Chart</a>
  <a class="nl" href="#census">Census</a>
  <a class="nl" href="#families">Certificates</a>
  <a class="nl" href="#solutions">Solutions</a>
  <a class="nl" href="#method">Method</a>
  <a class="nl" href="#reproduce">Reproduce</a>
  <a class="nl" href="{REPO}">GitHub</a>
</div></nav>
<main id="top">
<header>
  <span class="eyebrow">A certified census · CSS codes over 𝔽₂</span>
  <h1>The Strict-Transversal Gate Zoo</h1>
  <p class="standfirst">41 well-known CSS and quantum LDPC codes, four transversality
  classes — strict, fold, T-level, automorphism — one exact certificate per verdict.
  Every number on this page is computed and certified by the tool; proofs of absence are
  rank certificates, not searches that gave up.</p>
  <div class="stats">
    <div class="stat"><b>{len(DATA)}</b><span>codes</span></div>
    <div class="stat"><b>{FAMILY_COUNT}</b><span>families</span></div>
    <div class="stat"><b>{len(POSITIVE)}</b><span>strict gates</span></div>
    <div class="stat"><b>{FOLD_CERTIFIED}</b><span>fold gates</span></div>
    <div class="stat"><b>{T_CERTIFIED}</b><span>T-level gates</span></div>
    <div class="stat"><b>{BEST_EFF:.0f}</b><span>best kd²/n with gates:
      <a href="#{BEST_EFF_CODE['name']}">{BEST_EFF_CODE['name']} {nkd(BEST_EFF_CODE)}</a></span></div>
  </div>
  <p class="colophon count">every verdict machine-certified · method: Albert, arXiv:2608.05688 ·
  computed 2026-08-11 with <a href="{REPO}">qec-transversal</a></p>
</header>

<h2 id="definitions"><span class="no">§1</span>Six kinds of “transversal” — four decided here</h2>
<p class="narrow">The word “transversal” names many inequivalent gate classes in the
literature, differing in how far a single fault can spread. This zoo decides <b>four of
them</b> for every code: <b>strict</b> — one depth-one layer
U₁&nbsp;⊗&nbsp;…&nbsp;⊗&nbsp;U<sub>n</sub> of independent single-qubit Cliffords, no
two-qubit gates, no permutations (the strongest notion: a fault reaches nothing);
<b>T-level</b> — the same shape at the third hierarchy level, ∏ᵢ Tᵢ^tᵢ; <b>fold</b> —
single-qubit gates plus CZ across the pairs of a ZX-duality; and <b>automorphism</b> —
permutation gates from the code's symmetry group. The first two are decided by kernel
computations, complete outright; fold is complete per certified duality; automorphism
groups are exact for the given checks.</p>
<p class="narrow">When a paper says the toric or BB code “has transversal gates”, it means
one of the weaker rows below — which is why our strict verdict (“none”) and that folklore
are both right:</p>
<div class="tablewrap"><table class="defs">
<thead><tr><th>class</th><th>gate shape</th><th>one fault spreads to</th>
<th>example</th><th>status for qLDPC codes</th></tr></thead>
<tbody>
<tr class="ours"><td><b>strict</b> <span class="chip yes">decided</span></td>
<td>⊗ᵢ Uᵢ, one block, any single-qubit Clifford Uᵢ</td><td>nothing</td>
<td>S<sup>⊗7</sup> = S̄ on Steane; √Z layer = all-pairs CZ̄ on iceberg codes</td>
<td>empty for every qLDPC family (§4); 12 codes have gates (§5)</td></tr>
<tr class="ours"><td><b>T-level</b> <span class="chip yes">decided</span></td>
<td>⊗ᵢ Tᵢ^tᵢ, t ∈ ℤ₈ⁿ — the transversal T / CCZ family</td><td>nothing</td>
<td>T̄ on QRM [[15,1,3]]; CCZ̄ on the [[8,3,2]] cube code</td>
<td>only QRM15 and the cube code have one; every qLDPC code is Pauli-level</td></tr>
<tr class="ours"><td><b>fold</b> <span class="chip yes">per duality</span></td>
<td>single-qubit gates + CZ across the pairs of a ZX-duality fold</td>
<td>at most its fold partner</td>
<td>surface-code fold S; H-with-translation on toric; CZ/S fold gates on symmetric BB</td>
<td>28 codes have gates — every qLDPC family except GF(256) Kasai</td></tr>
<tr class="ours"><td><b>automorphism</b> <span class="chip yes">exact group</span></td>
<td>single-qubit layer + a qubit permutation from a code automorphism</td>
<td>nothing (relabeling), but needs physical routing</td>
<td>Swap<sub>x</sub>, Swap<sub>y</sub> lattice translations on BB codes</td>
<td>group order and logical action computed for every code (|Aut| column)</td></tr>
<tr><td>uniform transversal</td>
<td>U<sup>⊗n</sup>, the same gate on every qubit</td><td>nothing</td>
<td>H<sup>⊗n</sup> on self-dual codes</td>
<td>subclass of strict — covered; also empty for qLDPC codes</td></tr>
<tr><td>multi-block transversal (Eastin–Knill sense)</td>
<td>⊗ᵢ Uᵢ with Uᵢ acting on the i-th qubit of <em>every</em> block</td>
<td>nothing within a block</td>
<td>blockwise CNOT between two copies of any CSS code</td>
<td>always available; Eastin–Knill: never universal for d ≥ 2</td></tr>
<tr><td>depth-one two-local</td>
<td>any one layer of 1- and 2-qubit gates on a fixed qubit matching</td>
<td>at most its matching partner</td>
<td>gross-code CZ matchings (Albert, arXiv:2608.05688)</td>
<td>strictly larger than fold; this tool’s roadmap</td></tr>
<tr><td>constant-depth local circuit</td>
<td>any O(1)-depth geometrically local circuit</td>
<td>a constant-radius lightcone</td>
<td>—</td>
<td>Bravyi–König: capped at Clifford for 2D topological codes</td></tr>
</tbody></table></div>
<p class="narrow">One trust note covers the whole page: every positive verdict is
re-proved by the tool's own linear algebra; the only results resting on an external exact
tool are duality <em>non</em>-existence and automorphism-group completeness, which come from
BLISS canonical search on the given check set. Everything else is certificates all the way
down.</p>

<h2 id="chart"><span class="no">§2</span>The map</h2>
<p class="narrow">Each point is a code at ([[n, k]], log–log). Color is the strongest gate
class the code supports — <b>green</b>: strict gates; <b>blue</b>: fold gates only;
<b>grey</b>: none — and a <b>gold ring</b> marks a T-level gate. Hover any point for its
verdict; click to jump to its certificate. Three patterns to see: green never touches the
LDPC families, the gold rings never leave the small algebraic codes, and the single grey
dot is the GF(256) Kasai code, whose randomized labels provably destroy every symmetry.</p>
<div class="chartcard">
{scatter_svg()}
<div class="legend">
  <span><span class="dot strict"></span>strict gates</span>
  <span><span class="dot fold"></span>fold gates only</span>
  <span><span class="dot none"></span>no gates in any class</span>
  <span><span class="ring"></span>T-level gate</span>
  <span><span class="star">★</span>&nbsp;full logical Clifford group</span>
</div>
</div>

<h2 id="census"><span class="no">§3</span>The census</h2>
<p class="narrow">Search accepts free text (<code>bicycle</code>, <code>kasai</code>) and
filters: <code>n&gt;=100</code>, <code>k&gt;12</code>, <code>d&gt;=10</code>, <code>eff&gt;=10</code>
(eff = kd²/n), <code>lvl&gt;=3</code> (diag hierarchy level), <code>aut&gt;=100</code>,
<code>gates:yes</code>, <code>gates:no</code> — combine them with spaces. Click a column
header to sort; click a code name for its certificate.</p>
<div class="filter">
  <input id="q" type="search" placeholder="filter codes… e.g.  bicycle n>=100 gates:no" aria-label="Filter codes">
  <p class="hint"><span id="rowcount"></span> · dim A<sub>Z</sub>/A<sub>X</sub> are the two
  parameter-space dimensions · logical group is the exact order of the generated group</p>
</div>
<div class="tablewrap"><table id="census">
<thead><tr>
<th data-sort="name">code<span class="arr"></span></th>
<th data-sort="n">[[n,k,d]]<span class="arr"></span></th>
<th data-sort="family">family<span class="arr"></span></th>
<th data-sort="rate">rate k/n<span class="arr"></span></th>
<th data-sort="eff" title="operational figure of merit; surface code ~ 1">kd²/n<span class="arr"></span></th>
<th data-sort="az">dim A<sub>Z</sub>/A<sub>X</sub><span class="arr"></span></th>
<th data-sort="order">logical group<span class="arr"></span></th>
<th data-sort="gates" title="strict single-qubit layers">strict gates?<span class="arr"></span></th>
<th data-sort="fold" title="diagonal fold layers + fold-Hadamard per certified ZX-duality; combined logical group order">fold gates?<span class="arr"></span></th>
<th data-sort="lvl" title="highest certified diagonal Clifford-hierarchy level (3 = transversal T/CCZ family)">diag level<span class="arr"></span></th>
<th data-sort="aut" title="exact Tanner-graph automorphism group order (given checks)">|Aut|<span class="arr"></span></th>
</tr></thead>
<tbody>
{census_rows}
</tbody></table></div>

<h2 id="families"><span class="no">§4</span>The qLDPC families: strict-class certificates</h2>
<p class="narrow">Click any entry to expand. Each shows two verdicts in one card: the
<b>strict</b> certificate — the constraint matrix reaches full rank n in both sectors, so by
the <a href="#method">completeness theorem</a> no strict layer acts as a nontrivial logical
gate — and the <b>fold</b> verdict for the same code, with its certified dualities and exact
gate-group order. The strict emptiness is folklore-expected (the bicycle-code literature
goes straight to fold constructions); the systematic certificates are new.</p>
{''.join(groups_html)}

<h2 id="solutions"><span class="no">§5</span>The codes with strict gates: exact solutions</h2>
<p class="narrow">The exact solutions. Each filled strip is a parameter vector: apply √Z
(or √X) on the filled qubits. These are the classical positive controls — self-dual or
dual-containing CSS codes — and none of them is LDPC: their gates come from algebraic
structure (doubly-even self-duality, Reed–Muller nesting) that check-sparsity destroys.</p>
<div class="callout narrow">
  <span class="eyebrow">So which code with gates is “best”? Three answers</span>
  <p><b>By encoding rate k/n alone:</b> the iceberg family [[2m,&thinsp;2m−2,&thinsp;2]] —
  <a href="#iceberg-12">iceberg-12 = [[12,10,2]]</a> holds the zoo record at rate 5/6, and the
  family reaches rate → 1. But distance is frozen at 2: these codes only <em>detect</em>
  errors. Raw rate is a misleading scoreboard.</p>
  <p><b>By the operational figure of merit kd²/n</b> (surface code ≈ 1, the qldpc-challenge
  metric): the middle Reed–Muller family wins decisively —
  <a href="#rm256">rm256 = [[256,70,16]]</a> scores <b>kd²/n = 70</b> and
  <a href="#rm64">rm64 = [[64,20,8]]</a> scores 20, versus 12 for the gross code, 13.5 for
  two-gross, and ≤ 24.5 for bb756. In this family kd²/n = C(m,&thinsp;m/2) is unbounded — but
  the checks are dense (weight 32–256 at n = 256), so the LDPC property is the price of
  admission.</p>
  <p><b>Within LDPC and growing distance:</b> nothing <em>strict</em> — every qLDPC family
  here is certified strictly gate-free. The certified fold layer softens this only a little:
  the fold groups we certify are small (order 6, or 48 on symmetric BB codes) — global
  S̄/H̄-type actions, far short of the full logical Clifford group. High rate, growing
  distance, sparse checks, and a <em>large</em> transversal-class gate group in one code
  remains open territory.</p>
</div>
{positives_html}

<div class="narrow">
<div class="callout narrow">
  <span class="eyebrow">Beyond Clifford: certified level-3 gates</span>
  Two codes in the zoo carry a certified gate <em>outside</em> the Clifford group, decided
  exactly by the ℤ₈ kernel of the transversal-T family: <a href="#qrm15">qrm15</a> — the
  all-T layer implements logical T̄ (the textbook transversal-T code) — and
  <a href="#cube-832">cube-832</a> — the all-T layer implements logical CCZ̄ on its three
  logical qubits. Every other code, including every qLDPC family, is certified to top out
  at Clifford level (or below) for strict diagonal layers.
</div>

<h2 id="method"><span class="no">§6</span>The method, and why a trivial kernel is a proof</h2>
<p>A <em>strict-transversal</em> gate applies one single-qubit Clifford to each physical
qubit — no two-qubit gates, no permutations — so faults cannot spread inside a block.
Write C<sub>X</sub>, C<sub>Z</sub> for the row spans of the check matrices and ⊙ for the
coordinatewise product. A phase layer
U<sub>Z</sub>(a) = ∏<sub>i</sub> √Z<sub>i</sub><sup>a<sub>i</sub></sup> conjugates
X<sub>v</sub> to X<sub>v</sub>&thinsp;Z<sub>v⊙a</sub>, so it preserves the stabilizer
exactly when</p>
<div class="math">
<p>A<sub>Z</sub> = {{ a ∈ 𝔽₂ⁿ : a ⊙ C<sub>X</sub> ⊆ C<sub>Z</sub> }},&emsp;
A<sub>X</sub> = {{ b ∈ 𝔽₂ⁿ : b ⊙ C<sub>Z</sub> ⊆ C<sub>X</sub> }}.</p>
<p>For each check x of H<sub>X</sub>, the condition a ⊙ x ∈ C<sub>Z</sub> is the linear
system Q<sub>Z</sub>&thinsp;diag(x)&thinsp;aᵀ = 0 with Q<sub>Z</sub> spanning
ker H<sub>Z</sub>. Stacking all checks:&ensp;<b>A<sub>Z</sub> = ker M<sub>Z</sub></b>.
So <b>rank M<sub>Z</sub> = n</b> proves the kernel is {{0}}.</p>
</div>
<p>Diagonal layers are not the whole story — a transversal gate could mix X and Z (Hadamards,
a different Clifford per qubit). The completeness theorem (Albert, arXiv:2608.05688) closes
that gap: every strict-transversal Clifford factors as</p>
<div class="math"><p>g = H(t)&thinsp;U<sub>Z</sub>(q)&thinsp;U<sub>X</sub>(p),&emsp;
t ∈ A<sub>Z</sub> ∩ A<sub>X</sub>,&ensp;q ∈ A<sub>Z</sub>,&ensp;p ∈ A<sub>X</sub>,&ensp;q ⊙ p = 0,</p>
<p>with H(t) = U<sub>Z</sub>(t)U<sub>X</sub>(t)U<sub>Z</sub>(t). Hence
A<sub>Z</sub> = A<sub>X</sub> = {{0}} ⇒ the only strict-transversal gates are Paulis.</p></div>
<div class="callout">
  <span class="eyebrow">Exhaustive validation</span>
  We did not take the theorem on faith. For 46 codes with n ≤ 8 we enumerated
  <b>all 6ⁿ assignments</b> of arbitrary single-qubit Cliffords (up to 1,679,616 layers for
  the [[8,2,2]] toric code), kept those preserving the stabilizer, and compared. In every
  case the brute-force group, the group generated from A<sub>Z</sub>/A<sub>X</sub> alone, and
  the counting formula #{{(t,q,p) : q⊙p = 0}} agreed <b>exactly</b>. For the toric code the
  brute force found only the identity — a trivial kernel is a genuine nonexistence proof, not
  a blind spot of the diagonal search.
</div>
<p><b>Scope.</b> This zoo certifies the strict class only. <em>Fold-transversal</em> gates —
which add qubit permutations and two-qubit CZ layers (Breuckmann–Burton, arXiv:2202.06647;
Eberhardt–Steffan, arXiv:2407.03973) — are a strictly larger class, and that is where the
useful Clifford gates of bicycle codes actually live.</p>

<h2 id="reproduce"><span class="no">§7</span>Reproduce every number</h2>
<pre>git clone {REPO}
pip install -e .
qec-transversal list-codes
qec-transversal analyze --code gross          # any registry name
qec-transversal generate two-gross -o bb.json # export H_X, H_Z</pre>
<p>Every report carries a certificate block: CSS orthogonality, canonical logical pairing,
nullspace verification, per-generator symplectic checks, and the group-order cross-check
(Schreier–Sims against explicit enumeration).</p>

<footer>
<p><b>Sources.</b></p>
<ul>
<li>V. V. Albert, “Beyond transversality: structure of Clifford circuits for CSS codes,” arXiv:2608.05688 — the parameter-code method and normal form.</li>
<li>S. Bravyi et al., “High-threshold and low-overhead fault-tolerant quantum memory,” Nature 627, 778 (2024), arXiv:2308.07915.</li>
<li>J. N. Eberhardt, V. Steffan, “Logical operators and fold-transversal gates of bivariate bicycle codes,” arXiv:2407.03973.</li>
<li>P. Panteleev, G. Kalachev, “Degenerate quantum LDPC codes with good finite length performance,” Quantum 5, 585 (2021), arXiv:1904.02703.</li>
<li>D. Komoto, K. Kasai, “Quantum error correction near the coding theoretical bound,” npj Quantum Inf. 11, 154 (2025), arXiv:2412.21171.</li>
<li>L. Pecorari et al., “High-rate quantum LDPC codes for long-range-connected neutral atom registers,” Nat. Commun. 16, 1111 (2025), arXiv:2404.13010.</li>
<li>N. P. Breuckmann, S. Burton, “Fold-transversal Clifford gates for quantum codes,” Quantum 8, 1372 (2024), arXiv:2202.06647.</li>
</ul>
<p>All {len(DATA)} verdicts certified by <a href="{REPO}">qec-transversal</a> on 2026-08-11; analysis
wall-time totals under 15 seconds. Distances marked ≤ are published upper bounds. The Kasai
GF(256) instance uses the canonical separable label assignment, hence k = 800 (the paper's
randomized labels give 784). Site generated by <code>docs/zoo/make_zoo.py</code>.</p>
</footer>
</div>
</main>
<script>{JS}</script>
</body>
</html>
"""

path = HERE.parent / "index.html"
path.write_text(html)
print("wrote", path, len(html), "bytes")
