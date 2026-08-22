"""Physical gate ansaetze: what class of circuits is being searched.

Every module here answers the same question for a different ansatz -- *which
physical transformations of this shape preserve the code, and what do they do
logically?*  Two are general (arbitrary stabilizer codes, any prescribed
partition); the rest are specialised fast paths that exploit extra structure.

============================  ========================  =====================
module                        ansatz                    code class
============================  ========================  =====================
:mod:`.strict`                one Clifford per qubit    any stabilizer code
:mod:`.partition`             one Clifford per cell     any stabilizer code
:mod:`.strict_css`            diagonal sqrt(Z)/sqrt(X)  CSS
:mod:`.matching`              diagonal layer + fold H   CSS + an involution
:mod:`.twofold`               matching layers, sampled  CSS
:mod:`.permutation`           qubit permutations        CSS (Tanner graph)
:mod:`.codeword_permutation`  qubit permutations        CSS (row space)
:mod:`.monomial`              permutation x local Cliff any stabilizer code
============================  ========================  =====================

:mod:`.discovery` and :mod:`.dualities` are *candidate generators*, not
solvers: they propose involutions and permutations that the certified
analyses above then accept or reject.
"""
