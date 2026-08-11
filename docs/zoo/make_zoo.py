# Generates the Strict-Transversal Gate Zoo (docs/index.html) from zoo_data.json.
import json
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "zoo_data.json").read_text())
BY = {d["name"]: d for d in DATA}

# Human definitions per code (kept in sync with src/qec_transversal/codes.py).
DEFS = {
    "steane": "H<sub>X</sub> = H<sub>Z</sub> = parity checks of the [7,4,3] Hamming code. Self-dual, doubly even.",
    "c4-22": "H<sub>X</sub> = H<sub>Z</sub> = [1111]. The smallest error-detecting CSS code.",
    "c6-22": "H<sub>X</sub> = H<sub>Z</sub> = {111100, 110011}.",
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

POSITIVE = ["steane", "c4-22", "c6-22", "qrm15", "tesseract", "rm64", "grid-4x6", "grid-6x8"]


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
    return f'<span class="bits" role="img" aria-label="binary vector, weight {len(support)} of {n}">{"".join(cells)}</span>'


def matrix2(m):
    rows = ["&thinsp;".join(str(v) for v in row) for row in m]
    return '<span class="mat">(' + "&nbsp;;&nbsp;".join(rows) + ")</span>"


def trivial_entry(name):
    d = BY[name]
    assert d["dim_AZ"] == 0 and d["dim_AX"] == 0
    assert d["rank_MZ"] == d["n"] and d["rank_MX"] == d["n"]
    return f"""
<article class="entry" id="{name}">
  <header>
    <h4>{escape(name)} <span class="nkd">{nkd(d)}</span></h4>
    <span class="chip none">no transversal gate</span>
  </header>
  <p class="def">{DEFS[name]}</p>
  <p class="cert"><b>Certificate.</b>
    rank&nbsp;M<sub>Z</sub> = {d['rank_MZ']} = n and rank&nbsp;M<sub>X</sub> = {d['rank_MX']} = n,
    so A<sub>Z</sub> = ker&nbsp;M<sub>Z</sub> = {{0}} and A<sub>X</sub> = ker&nbsp;M<sub>X</sub> = {{0}}.
    By the completeness theorem the strict-transversal group is exactly the Pauli group.
    <span class="t">verified in {d['seconds']:.2f}&thinsp;s</span></p>
</article>"""


def positive_entry(name, extra=""):
    d = BY[name]
    n = d["n"]
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
    full = " = full logical Clifford group Sp(2k,2) mod Paulis" if d["is_full"] else ""
    return f"""
<article class="entry has" id="{name}">
  <header>
    <h4>{escape(name)} <span class="nkd">{nkd(d)}</span></h4>
    <span class="chip yes">transversal gates exist</span>
  </header>
  <p class="def">{DEFS[name]}</p>
  <div class="gens">{''.join(gens_html)}</div>
  <p class="cert"><b>Exact solution.</b>
    dim&nbsp;A<sub>Z</sub> = {d['dim_AZ']} (rank&nbsp;M<sub>Z</sub> = {d['rank_MZ']} of n = {n}),
    dim&nbsp;A<sub>X</sub> = {d['dim_AX']}.
    Logical group order {order}{full}.{extra}</p>
</article>"""


def census_row(name, kind):
    d = BY[name]
    chip = '<span class="chip yes">yes</span>' if kind else '<span class="chip none">none</span>'
    order = d["order"]
    return (f'<tr><td><a href="#{name}">{escape(name)}</a></td>'
            f'<td class="mono">{nkd(d)}</td><td>{escape(d["family"])}</td>'
            f'<td class="num">{d["dim_AZ"]} / {d["dim_AX"]}</td>'
            f'<td class="num">{order}</td><td>{chip}</td></tr>')


groups_html = []
for title, src, names in FAMILY_GROUPS:
    entries = "".join(trivial_entry(nm) for nm in names)
    groups_html.append(f"""
<section class="famgroup">
  <h3>{escape(title)}</h3>
  <p class="src">{src}</p>
  {entries}
</section>""")

qrm_extra = (" Four of the five √Z layers act as the logical identity (they lie on stabilizer-"
             "supported patterns); the fifth is the logical phase gate S̄ inherited from the "
             "code's transversal T.")
positives_html = "".join(
    positive_entry(nm, qrm_extra if nm == "qrm15" else "") for nm in POSITIVE
)

census_neg = "".join(census_row(nm, False) for _, _, names in FAMILY_GROUPS for nm in names)
census_pos = "".join(census_row(nm, True) for nm in POSITIVE)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Strict-Transversal Gate Zoo</title>
<meta name="description" content="A certified census of strict-transversal Clifford gates
across 37 well-known quantum LDPC and CSS codes: exact nonexistence certificates and exact
gate solutions, computed with qec-transversal.">
<style>
:root {{
  --paper:#FAFAF9; --ink:#1C2321; --muted:#5D6660; --rule:#DCE0DB;
  --accent:#17604E; --accent-ink:#0E4A3B; --chipno-bg:#EEF0ED; --chipno-tx:#5D6660;
  --chipyes-bg:#E1EEE9; --entry:#FFFFFF; --code-bg:#F1F3F0; --bit0:#E3E6E1; --bit1:#17604E;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#121614; --ink:#E4E8E4; --muted:#98A29B; --rule:#2A312C;
    --accent:#4CAF94; --accent-ink:#7CCBB4; --chipno-bg:#1D2320; --chipno-tx:#98A29B;
    --chipyes-bg:#173B30; --entry:#181D1A; --code-bg:#1D2320; --bit0:#2A312C; --bit1:#4CAF94;
  }}
}}
:root[data-theme="dark"] {{
  --paper:#121614; --ink:#E4E8E4; --muted:#98A29B; --rule:#2A312C;
  --accent:#4CAF94; --accent-ink:#7CCBB4; --chipno-bg:#1D2320; --chipno-tx:#98A29B;
  --chipyes-bg:#173B30; --entry:#181D1A; --code-bg:#1D2320; --bit0:#2A312C; --bit1:#4CAF94;
}}
* {{ box-sizing:border-box; }}
body {{
  background:var(--paper); color:var(--ink); margin:0;
  font:17px/1.62 "Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
}}
main {{ max-width:76ch; margin:0 auto; padding:3.5rem 1.25rem 5rem; }}
.mono, code, .mat, .nkd, .cert, table, .bits, .glabel, .gmeta {{
  font-family: ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}}
header.mast {{ border-bottom:3px double var(--rule); padding-bottom:1.6rem; margin-bottom:2.2rem; }}
.eyebrow {{ font-family:ui-monospace,Menlo,monospace; font-size:.72rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--accent); }}
h1 {{ font-size:2.5rem; line-height:1.12; margin:.35rem 0 .7rem; font-weight:600; text-wrap:balance; }}
.standfirst {{ font-size:1.12rem; color:var(--muted); margin:0 0 1rem; max-width:62ch; }}
.colophon {{ font-family:ui-monospace,Menlo,monospace; font-size:.78rem; color:var(--muted); }}
.colophon b {{ color:var(--ink); font-weight:600; }}
h2 {{ font-size:1.45rem; margin:2.8rem 0 .8rem; text-wrap:balance; }}
h2 .no {{ color:var(--accent); font-family:ui-monospace,Menlo,monospace; font-size:.95rem;
  vertical-align:.18em; margin-right:.45em; }}
h3 {{ font-size:1.12rem; margin:2rem 0 .1rem; }}
p {{ margin:.65rem 0; }}
.math {{ background:var(--code-bg); border-left:3px solid var(--accent); padding:.8rem 1.1rem;
  margin:1rem 0; font-size:.98rem; overflow-x:auto; }}
.math p {{ margin:.4rem 0; }}
var {{ font-style:italic; }}
.src {{ font-size:.8rem; font-family:ui-monospace,Menlo,monospace; color:var(--muted); margin:.1rem 0 .9rem; }}
.tablewrap {{ overflow-x:auto; margin:1.2rem 0; }}
table {{ border-collapse:collapse; font-size:.82rem; width:100%; min-width:640px; }}
th {{ text-align:left; font-weight:600; letter-spacing:.06em; text-transform:uppercase;
  font-size:.68rem; color:var(--muted); border-bottom:2px solid var(--rule); padding:.45rem .7rem .45rem 0; }}
td {{ border-bottom:1px solid var(--rule); padding:.42rem .7rem .42rem 0; }}
td.num {{ font-variant-numeric:tabular-nums; }}
td a {{ color:var(--accent-ink); text-decoration:none; }}
td a:hover, td a:focus-visible {{ text-decoration:underline; }}
.groupline td {{ border-bottom:2px solid var(--rule); color:var(--muted); font-size:.72rem;
  letter-spacing:.1em; text-transform:uppercase; padding-top:1.1rem; }}
.chip {{ display:inline-block; font-family:ui-monospace,Menlo,monospace; font-size:.68rem;
  padding:.1rem .55rem; border-radius:2px; white-space:nowrap; }}
.chip.none {{ background:var(--chipno-bg); color:var(--chipno-tx); }}
.chip.yes  {{ background:var(--chipyes-bg); color:var(--accent-ink); font-weight:700; }}
.entry {{ background:var(--entry); border:1px solid var(--rule); border-radius:3px;
  padding:.9rem 1.1rem; margin:.8rem 0; }}
.entry.has {{ border-left:4px solid var(--accent); }}
.entry header {{ display:flex; align-items:baseline; justify-content:space-between; gap:1rem; flex-wrap:wrap; }}
.entry h4 {{ margin:0; font-size:1rem; }}
.nkd {{ color:var(--muted); font-size:.85rem; font-weight:400; margin-left:.35em; }}
.def {{ font-size:.95rem; margin:.45rem 0; }}
.cert {{ font-size:.8rem; color:var(--muted); margin:.45rem 0 0; }}
.cert b {{ color:var(--ink); }}
.t {{ float:right; opacity:.8; }}
.gens {{ display:flex; flex-direction:column; gap:.45rem; margin:.6rem 0; }}
.gen {{ font-size:.8rem; overflow-x:auto; }}
.glabel {{ color:var(--accent-ink); font-weight:700; margin-right:.4em; }}
.gmeta {{ color:var(--muted); margin-left:.5em; }}
.bits {{ display:inline-block; line-height:14px; vertical-align:middle; }}
.bits i {{ display:inline-block; width:10px; height:10px; margin:1px; border-radius:1px; }}
.bits .b0 {{ background:var(--bit0); }}
.bits .b1 {{ background:var(--bit1); }}
.mat {{ white-space:nowrap; }}
.callout {{ border:1px solid var(--rule); border-radius:3px; padding:.9rem 1.1rem; margin:1.2rem 0;
  background:var(--entry); font-size:.95rem; }}
.callout .eyebrow {{ display:block; margin-bottom:.3rem; }}
pre {{ background:var(--code-bg); padding:.8rem 1rem; overflow-x:auto; font-size:.82rem;
  border-radius:3px; }}
footer {{ margin-top:3.5rem; border-top:1px solid var(--rule); padding-top:1rem;
  font-size:.8rem; color:var(--muted); }}
footer ul {{ padding-left:1.2rem; margin:.4rem 0; }}
a {{ color:var(--accent-ink); }}
@media (prefers-reduced-motion: reduce) {{ * {{ scroll-behavior:auto; }} }}
</style>
<main>
<header class="mast">
  <span class="eyebrow">A certified census · CSS codes over 𝔽₂</span>
  <h1>The Strict-Transversal Gate Zoo</h1>
  <p class="standfirst">Every well-known quantum LDPC code, one exact linear-algebra
  certificate each: which codes admit a depth-one layer of single-qubit Clifford gates acting
  as a nontrivial logical gate — and the proof that most admit none.</p>
  <p class="colophon"><b>37 codes</b> · 11 families · every verdict machine-certified ·
  computed 2026-08-10 with <b>qec-transversal</b> (method: Albert, arXiv:2608.05688)</p>
</header>

<h2><span class="no">§1</span>The question</h2>
<p>A <em>strict-transversal</em> gate on one code block applies one single-qubit Clifford to
each physical qubit — no two-qubit gates, no qubit permutations. Faults cannot spread inside
the block, so these are the cheapest fault-tolerant logical gates a code can have. The
question for each code: <em>does any such layer act as a nontrivial logical Clifford?</em></p>

<h2><span class="no">§2</span>The method, in four lines</h2>
<p>Write C<sub>X</sub>, C<sub>Z</sub> for the row spans of the check matrices H<sub>X</sub>,
H<sub>Z</sub>, and ⊙ for the coordinatewise product of binary vectors. A layer of phase gates
U<sub>Z</sub>(a) = ∏<sub>i</sub> √Z<sub>i</sub><sup>a<sub>i</sub></sup> conjugates an X-type
stabilizer X<sub>v</sub> to X<sub>v</sub>&thinsp;Z<sub>v⊙a</sub>. It preserves the stabilizer
group exactly when every such Z-residue is again a stabilizer:</p>
<div class="math">
<p>A<sub>Z</sub> = {{ a ∈ 𝔽₂ⁿ : a ⊙ C<sub>X</sub> ⊆ C<sub>Z</sub> }},&emsp;
A<sub>X</sub> = {{ b ∈ 𝔽₂ⁿ : b ⊙ C<sub>Z</sub> ⊆ C<sub>X</sub> }}.</p>
<p>For each check x of H<sub>X</sub>, the condition a ⊙ x ∈ C<sub>Z</sub> is the linear system
Q<sub>Z</sub>&thinsp;diag(x)&thinsp;aᵀ = 0, where Q<sub>Z</sub> spans ker H<sub>Z</sub>.
Stacking all checks gives one matrix M<sub>Z</sub> with&ensp;<b>A<sub>Z</sub> = ker M<sub>Z</sub></b>.</p>
</div>
<p>So the entire search is a nullspace computation over 𝔽₂ — exact, fast, and certifiable:
<b>rank M<sub>Z</sub> = n</b> is a proof that the kernel is {{0}}. The same construction with
X and Z exchanged gives A<sub>X</sub>.</p>

<h2><span class="no">§3</span>Why a trivial kernel is a full impossibility proof</h2>
<p>A skeptic should object: the kernels above only cover <em>diagonal</em> layers
(√Z- and √X-type). What about layers mixing X and Z — a Hadamard on some qubits, a different
Clifford on every qubit? The answer is a completeness theorem
(Albert, arXiv:2608.05688): every strict-transversal Clifford factors as</p>
<div class="math"><p>g = H(t)&thinsp;U<sub>Z</sub>(q)&thinsp;U<sub>X</sub>(p),&emsp;
t ∈ A<sub>Z</sub> ∩ A<sub>X</sub>,&ensp;q ∈ A<sub>Z</sub>,&ensp;p ∈ A<sub>X</sub>,&ensp;q ⊙ p = 0,</p>
<p>where H(t) = U<sub>Z</sub>(t)U<sub>X</sub>(t)U<sub>Z</sub>(t) is a subset-Hadamard.
Hence A<sub>Z</sub> = A<sub>X</sub> = {{0}} &nbsp;⇒&nbsp; the only strict-transversal gates are Pauli operators.</p></div>
<div class="callout">
  <span class="eyebrow">Exhaustive validation</span>
  We did not take the theorem on faith. For 46 codes with n ≤ 8 we enumerated
  <b>all 6ⁿ assignments</b> of arbitrary single-qubit Cliffords (up to 1,679,616 layers for the
  [[8,2,2]] toric code), kept those preserving the stabilizer, and compared. In every case the
  brute-force group, the group generated from A<sub>Z</sub>/A<sub>X</sub> alone, and the
  counting formula #{{(t,q,p) : q⊙p = 0}} agreed <b>exactly</b>. For the toric code the brute
  force found only the identity — the trivial kernel is a genuine nonexistence proof, not a
  blind spot of the diagonal search.
</div>
<p><b>Scope.</b> This zoo certifies the strict class only. <em>Fold-transversal</em> gates —
which add qubit permutations and two-qubit CZ layers (Breuckmann–Burton, arXiv:2202.06647;
Eberhardt–Steffan, arXiv:2407.03973) — are a strictly larger class, and that is where the
useful Clifford gates of bicycle codes actually live.</p>

<h2><span class="no">§4</span>The census</h2>
<p>dim A<sub>Z</sub> / dim A<sub>X</sub> are the dimensions of the two parameter spaces; the
logical group is the exact order of the group of logical actions they generate
(Schreier–Sims, cross-checked by explicit enumeration).</p>
<div class="tablewrap"><table>
<thead><tr><th>code</th><th>[[n,k,d]]</th><th>family</th><th>dim A<sub>Z</sub>/A<sub>X</sub></th>
<th>logical group</th><th>gates?</th></tr></thead>
<tbody>
<tr class="groupline"><td colspan="6">quantum LDPC codes — all trivial</td></tr>
{census_neg}
<tr class="groupline"><td colspan="6">positive controls — gates exist</td></tr>
{census_pos}
</tbody></table></div>

<h2><span class="no">§5</span>The qLDPC families: nonexistence certificates</h2>
<p>Each entry states the code, then the certificate: the constraint matrix reaches full rank
n in both sectors, so both kernels are {{0}} and — by §3 — no strict-transversal layer acts as
a nontrivial logical gate. This is folklore-expected (the bicycle-code literature goes
straight to fold-transversal constructions) but, to our knowledge, certified systematically
here for the first time.</p>
{''.join(groups_html)}

<h2><span class="no">§6</span>The codes that do have transversal gates</h2>
<p>The exact solutions. Each filled strip is a parameter vector: apply √Z (or √X) on the
filled qubits. These are the classical positive controls — self-dual or dual-containing CSS
codes — and none of them is LDPC: their gates come from algebraic structure
(doubly-even self-duality, Reed–Muller nesting) that check-sparsity destroys.</p>
{positives_html}

<h2><span class="no">§7</span>Reproduce every number</h2>
<pre>pip install -e .        # github: qec-transversal (this project)
qec-transversal list-codes
qec-transversal analyze --code gross          # any registry name
qec-transversal generate two-gross -o bb.json # export H_X, H_Z</pre>
<p>Every report carries a certificate block: CSS orthogonality, canonical logical pairing,
nullspace verification, per-generator symplectic checks, and the group-order cross-check.</p>

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
<p>All 37 verdicts certified by qec-transversal on 2026-08-10; analysis wall-time totals under
15 seconds. Distances d marked ≤ are published upper bounds. The Kasai GF(256) instance uses
the canonical separable label assignment, hence k = 800 (the paper's randomized labels give 784).</p>
</footer>
</main>
</body>
</html>
"""

path = HERE.parent / "index.html"
path.write_text(html)
print("wrote", path, len(html), "bytes")
