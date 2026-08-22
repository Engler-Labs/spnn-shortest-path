"""Exact combinatorial counters for the SPNN discrimination threshold (Part B/C).

Three exact counts (no sampling, no estimation) back the ``CNT-*`` / ``POP-*`` /
``T-CREQDIST`` claims:

  * ``count_paths_by_length``  -- simple s->t paths by hop count, and the minimum
    path weight at each hop count.  Depth-limited DFS; exact by exhaustion.
  * ``count_matchings``        -- the matching generating polynomial coefficients
    m(G,k) = the number of sets of k pairwise vertex-disjoint edges, up to k=kmax.
    Frontier dynamic program over a vertex ordering: process vertices in order,
    the DP state being the set of not-yet-processed vertices already CLAIMED by a
    chosen edge.  At most kmax vertices can be claimed at once, so the state space
    is bounded by the subsets of the ordering's boundary of size <= kmax.  Exact.
  * ``scatter_counts``         -- exact scatter multiplicities at a matched size on
    the AUGMENTED graph, decomposed by auxiliary-edge usage (built on m(G,k)).

⚠️ EXACTNESS OF ``count_matchings`` (stage 26 -> stage 27).  The polynomial is
carried packed into ONE Python integer with fixed-width coefficient fields.  Until
stage 27 that width was hard-coded ``DIG = 96``, which is exact only while every
coefficient stays below 2^96; above it a coefficient carries into the next field
and the top one -- the one #scatter is read from -- is MASKED.  It did not raise or
flag, it returned a plausible integer, and it was always TOO SMALL, which makes the
required ``c`` look smaller (easier) than it truly is.  Stage 26 measured the damage
against an independent unpacked DP: grid k=12 was 128x too small, mtn_knn N=180
4,642x too small.  The width is now SIZED FROM THE INSTANCE from the provable bound
m(G,k) <= C(|E|,k), a too-narrow explicit width raises ``CoefficientOverflow``
instead of returning a number, and every call reports its field width and its
largest coefficient's bit length in ``stats`` (``max_coeff_bits`` per row).

Provenance
----------
``adjacency``, ``count_paths_by_length``, ``count_matchings`` (packed, auto-sized
``DIG``, ``CoefficientOverflow``) and their helpers are extracted verbatim from an
internal reference harness at a specific source revision -- the stage-27 fix that
sizes the coefficient field from the instance (a later revision of that source
reverted it). ``scatter_counts`` is extracted verbatim from another internal
harness (unchanged across those source revisions), with its ``count_matchings``
call repointed at the copy in this module. A thread-pinning entry-point import (an
OpenBLAS oversubscription guard) and an unused ``numpy`` import are dropped --
thread pinning is a harness concern, not the library's.
"""

import math
from collections import defaultdict, deque


# ----------------------------------------------------------------------
def adjacency(n, edges, weights=None):
    adj = [[] for _ in range(n)]
    for i in range(len(edges)):
        u, v = int(edges[i][0]), int(edges[i][1])
        w = float(weights[i]) if weights is not None else 1.0
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


# ----------------------------------------------------------------------
def count_paths_by_length(n, edges, weights, s, t, Lmax):
    """EXACT count of simple s->t paths with exactly L edges, for L = 1..Lmax.

    Also returns, per L, the MINIMUM total weight over those paths and one
    witness path attaining it.  Plain depth-limited DFS over simple paths --
    exhaustive, hence exact; the only cost is time.

    Returns (counts, minw, witness) each indexed 0..Lmax.
    """
    adj = adjacency(n, edges, weights)
    counts = [0] * (Lmax + 1)
    minw = [float("inf")] * (Lmax + 1)
    witness = [None] * (Lmax + 1)
    on = [False] * n
    stack_path = [s]
    on[s] = True

    def rec(v, depth, wsum):
        if v == t:
            counts[depth] += 1
            if wsum < minw[depth]:
                minw[depth] = wsum
                witness[depth] = list(stack_path)
            # a simple path that has reached t cannot be extended THROUGH t
            return
        if depth == Lmax:
            return
        for u, w in adj[v]:
            if on[u]:
                continue
            on[u] = True
            stack_path.append(u)
            rec(u, depth + 1, wsum + w)
            stack_path.pop()
            on[u] = False

    rec(s, 0, 0.0)
    return counts, minw, witness


# ----------------------------------------------------------------------
def _boundary_order(n, adj):
    """Vertex ordering minimising the running frontier, over BFS starts + greedy.

    The frontier at step i is {u not yet processed : u adjacent to some processed
    vertex}; the DP state space is bounded by subsets of it.  Cheap heuristic
    search: BFS from every vertex, plus a greedy min-frontier-growth ordering
    from every vertex; keep the ordering with the smallest maximum frontier.
    """
    best, best_w = None, None
    for start in range(n):
        # BFS
        order, seen = [], [False] * n
        dq = deque([start])
        seen[start] = True
        while dq:
            v = dq.popleft()
            order.append(v)
            for u, _ in adj[v]:
                if not seen[u]:
                    seen[u] = True
                    dq.append(u)
        for v in range(n):
            if not seen[v]:
                seen[v] = True
                order.append(v)
        w = _max_frontier(n, adj, order)
        if best_w is None or w < best_w:
            best, best_w = order, w
        # greedy: next vertex = the one minimising the resulting frontier
        order, placed = [], [False] * n
        frontier = set()
        cur = start
        for _ in range(n):
            placed[cur] = True
            order.append(cur)
            frontier.discard(cur)
            for u, _ in adj[cur]:
                if not placed[u]:
                    frontier.add(u)
            if len(order) == n:
                break
            cands = frontier if frontier else [v for v in range(n) if not placed[v]]
            cur = min(cands, key=lambda v: sum(
                1 for u, _ in adj[v] if not placed[u] and u not in frontier))
        w = _max_frontier(n, adj, order)
        if w < best_w:
            best, best_w = order, w
    return best, best_w


def _max_frontier(n, adj, order):
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i
    touched = [False] * n
    frontier = set()
    mx = 0
    for i, v in enumerate(order):
        frontier.discard(v)
        for u, _ in adj[v]:
            if pos[u] > i:
                frontier.add(u)
        mx = max(mx, len(frontier))
    return mx


# ----------------------------------------------------------------------
class StateBlowup(RuntimeError):
    """The DP state space exceeded the cap under the supplied vertex ordering."""


class CoefficientOverflow(RuntimeError):
    """A packed coefficient field is (or provably could be) too narrow.

    The packed representation is exact ONLY while every coefficient stays below
    2^DIG.  Above that a coefficient carries into the next field and the top one
    is masked -- silently, and always DOWNWARD, which in this project makes the
    required ``c`` look SMALLER than it truly is.  Rather than return a plausible
    integer, ``count_matchings`` raises this.
    """


# The historical fixed width.  Kept ONLY as the reference point for the headroom
# diagnostics and for the legacy-exactness verdict recorded in ``stats``; it is no
# longer what the DP runs at.
LEGACY_DIG = 96


def matching_coefficient_bound_bits(n_edges, kmax):
    """Bits provably sufficient for every coefficient of the k<=kmax DP.

    Every integer the frontier DP holds is, in its k-th field, a count of DISTINCT
    k-edge partial matchings of G -- injectively a set of k edges -- so it is at
    most C(|E|, k).  Taking the max over k = 0..kmax gives a width at which no
    field can ever carry, hence exactness by construction rather than by luck of
    instance size.  (The bound also covers the intermediate per-state values: at
    every step the k-matchings counted are partitioned across states, so no single
    state's k-field can exceed the total m(G,k) <= C(|E|,k).)
    """
    b = 1
    for k in range(0, kmax + 1):
        b = max(b, math.comb(n_edges, k).bit_length())
    return b


def count_matchings(n, edges, kmax, order=None, report=None, state_cap=None,
                    dig=None, slack=1):
    """EXACT m(G,k) for k = 0..kmax: the number of k-edge matchings of G.

    Frontier DP.  Vertices are processed in `order`.  The DP state is the SET of
    unprocessed vertices already claimed by a previously chosen matching edge;
    since each claim consumes one of at most kmax edges, |state| <= kmax always.
    The polynomial in k is carried as a single Python integer with fixed-width
    digits, so a transition is one shift and one add.

    EXACTNESS.  The coefficient field width is SIZED FROM THE INSTANCE -- see
    ``matching_coefficient_bound_bits`` -- so no field can overflow.  Passing an
    explicit ``dig`` narrower than that bound raises ``CoefficientOverflow``
    instead of returning the masked (too-small) answer the fixed ``DIG = 96`` used
    to return.  Every call also records, in ``stats``, the field width used, the
    largest coefficient's bit length, and the headroom to the legacy 2^96 line, so
    a future crossing is visible in the DATA and not only in an assertion.

    Returns (m, stats) with m[k] an exact Python int.
    """
    adj = adjacency(n, edges)
    nbrs = [sorted({u for u, _ in adj[v]}) for v in range(n)]
    n_distinct_edges = sum(len(nb) for nb in nbrs) // 2
    if order is None:
        order, _ = _boundary_order(n, nbrs and [[(u, 1.0) for u in nb] for nb in nbrs])
    pos = [0] * n
    for i, v in enumerate(order):
        pos[v] = i

    bound_bits = matching_coefficient_bound_bits(n_distinct_edges, kmax)
    if dig is None:
        DIG = bound_bits + int(slack)
    else:
        DIG = int(dig)
        if DIG < bound_bits:
            raise CoefficientOverflow(
                f"field width dig={DIG} is below the provable coefficient bound "
                f"{bound_bits} bits (max_k<={kmax} C({n_distinct_edges},k)); the packed "
                f"polynomial could overflow and be MASKED DOWNWARD.  Refusing to "
                f"return a possibly-wrong count -- pass dig=None to size it from "
                f"the instance.")
    MASK = (1 << DIG) - 1
    states = {0: 1}               # claimed-bitmask -> packed polynomial (k=0 term 1)
    max_states = 1

    for i, v in enumerate(order):
        bit_v = 1 << v
        new = defaultdict(int)
        for st, poly in states.items():
            if st & bit_v:
                new[st ^ bit_v] += poly
                continue
            new[st] += poly                       # v left unmatched
            for u in nbrs[v]:
                if pos[u] > i and not (st & (1 << u)):
                    # drop the k == kmax coefficient before shifting (it would
                    # overflow into a k > kmax term we do not want)
                    shifted = (poly - ((poly >> (DIG * kmax)) << (DIG * kmax))) << DIG
                    if shifted:
                        new[st | (1 << u)] += shifted
        states = dict(new)
        max_states = max(max_states, len(states))
        if state_cap is not None and max_states > state_cap:
            raise StateBlowup(
                f"DP states {max_states:,} exceeded cap {state_cap:,} at vertex "
                f"{i}/{n} under this ordering")
        if report is not None:
            report(i, v, len(states))

    total = 0
    for st, poly in states.items():
        assert st == 0, "a claimed vertex was never processed"
        total += poly
    m = [(total >> (DIG * k)) & MASK for k in range(kmax + 1)]

    # ---- per-instance guards: raise loudly, never return a plausible integer --
    # (1) nothing spilled above the top field.
    if total.bit_length() > DIG * kmax + DIG:
        raise CoefficientOverflow(
            f"packed polynomial occupies {total.bit_length()} bits, above the "
            f"{DIG * (kmax + 1)} bits of the kmax={kmax} field array: a coefficient "
            f"carried out of the top field at dig={DIG}")
    # (2) no coefficient reached the field width (equivalently: no field carried).
    max_coeff_bits = max((v.bit_length() for v in m), default=0)
    if max_coeff_bits >= DIG:
        raise CoefficientOverflow(
            f"coefficient of width {max_coeff_bits} bits meets or exceeds the field "
            f"width dig={DIG}; the packed result is NOT exact")
    # (3) every coefficient obeys the combinatorial bound it was sized from --
    #     a necessary condition that would catch corruption carried in from below.
    for k, v in enumerate(m):
        if v > math.comb(n_distinct_edges, k):
            raise CoefficientOverflow(
                f"m(G,{k}) = {v} exceeds C(|E|={n_distinct_edges},{k}) = "
                f"{math.comb(n_distinct_edges, k)}; the packed result is corrupt")
    # (4) cheap closed-form self-checks on the two coefficients we know outright.
    if m[0] != 1:
        raise CoefficientOverflow(f"m(G,0) = {m[0]}, must be 1")
    if kmax >= 1 and m[1] != n_distinct_edges:
        raise CoefficientOverflow(
            f"m(G,1) = {m[1]} but G has {n_distinct_edges} distinct edges")

    top_bits = m[kmax].bit_length()
    stats = dict(
        max_states=max_states,
        order_max_frontier=_max_frontier(
            n, [[(u, 1.0) for u in nb] for nb in nbrs], order),
        n_distinct_edges=n_distinct_edges,
        kmax=kmax,
        dig=DIG,
        coeff_bound_bits=bound_bits,
        slack=(DIG - bound_bits),
        top_coeff_bits=top_bits,
        max_coeff_bits=max_coeff_bits,
        argmax_coeff_k=max(range(kmax + 1), key=lambda k: m[k].bit_length()),
        legacy_dig=LEGACY_DIG,
        headroom_bits_to_legacy=LEGACY_DIG - max_coeff_bits,
        legacy_dig96_exact=bool(max_coeff_bits < LEGACY_DIG),
    )
    return m, stats


# ----------------------------------------------------------------------
def scatter_counts(n, edges, s, t, k):
    """Exact scatter multiplicities at matched size k on the AUGMENTED graph.

    Decomposed by how many of the two zero-weight auxiliary edges are used:
      N0 = k-matchings of G using neither  = m(G, k)
      N1 = using exactly one              = m(G-s, k-1) + m(G-t, k-1)
      N2 = using both                     = m(G-s-t, k-2)
    (G-x is G with vertex x and its incident edges deleted.)

    Microstate weights: a matching covering r REAL vertices carries 2^r equal-energy
    slot assignments; an auxiliary edge covers 1 real vertex, a real edge 2.
    """
    def sub(drop):
        keep = [(int(a), int(b)) for a, b in edges
                if int(a) not in drop and int(b) not in drop]
        return keep

    def mm(drop, kk):
        if kk < 0:
            return 0
        ee = sub(drop)
        if not ee:
            return 1 if kk == 0 else 0
        m, _ = count_matchings(n, ee, kk)
        return m[kk]

    N0 = mm(set(), k)
    N1 = mm({s}, k - 1) + mm({t}, k - 1)
    N2 = mm({s, t}, k - 2)
    edge_noaux = N0
    edge_aux = N0 + N1 + N2
    micro_noaux = N0 * (4 ** k)
    micro_aux = N0 * (4 ** k) + N1 * (2 ** (2 * k - 1)) + N2 * (2 ** (2 * k - 2))
    return dict(N0=N0, N1=N1, N2=N2, edge_noaux=edge_noaux, edge_aux=edge_aux,
                micro_noaux=micro_noaux, micro_aux=micro_aux)


# ----------------------------------------------------------------------
def brute_matchings(n, edges, kmax):
    """O(2^|E|)-ish reference for tiny graphs -- used to validate count_matchings."""
    from itertools import combinations
    m = [0] * (kmax + 1)
    m[0] = 1
    E = [(int(a), int(b)) for a, b in edges]
    for k in range(1, kmax + 1):
        for comb in combinations(E, k):
            vs = set()
            ok = True
            for u, v in comb:
                if u in vs or v in vs:
                    ok = False
                    break
                vs.add(u)
                vs.add(v)
            if ok:
                m[k] += 1
    return m


# ----------------------------------------------------------------------
def count_degree2_by_deficit(n, edges, s, t, L, order=None):
    """EXACT microstate-weighted N_d for the section V-D deficit grading.

    Counts the admissible matched-activity configurations on the AUGMENTED graph
    (original graph + terminals S,T + the two zero-weight auxiliary edges (S,s),(t,T)):
    the degree-<=2 subgraphs with exactly m = L+2 edges -- vertex-disjoint unions of
    paths and cycles.  The auxiliary edges are NOT forced: S and T
    may be inactive, because the S/T drive is a beta coupling in ``sim.energy``, not a
    static activation reward, so the S,T-inactive configurations have the SAME static
    energy and matched activity as the S,T-active ones and compete on equal footing.
    (An earlier version forced both aux edges and deg(s)=deg(t)<=1; that both-aux
    subset UNDER-counts the true scatter by ~2^(L+1) and is superseded here.)

    Each configuration carries its microstate multiplicity: a config covering r REAL
    vertices has 2^r equal-energy slot assignments, so it is weighted by
    ``2^(#real vertices covered)`` (an aux edge covers one real vertex, a real edge
    two).  ``Nd[d]`` is the summed weight, not a raw config count.

    Partitions by the DEFICIT ``d = k_p - 1``, where ``k_p`` = number of PATH
    components.  A max-degree-<=2 graph has exactly ``(#degree-1 vertices)/2`` path
    components; the degree-1 count INCLUDES the terminals S,T, so the full
    S-s-...-t-T path (S,T at degree 1) is d=0 and a pure disjoint-cycle set (no
    degree-1 vertices) is d=-1 -- the sector strictly below the path.  So d ranges
    over -1 .. m-1 and no connectivity tracking is needed: a frontier DP over vertex
    degrees (0/1/2), pruned by the m-edge budget, suffices.

    Counts accumulate in Python arbitrary-precision integers, so there is no packed
    field to overflow.  The d=m-1 (all-disjoint = a matching) term equals
    ``scatter_counts``' ``micro_aux`` -- a cross-check against the independent
    matching counter over all auxiliary-edge usages.

    Returns ``(Nd, stats)`` with ``Nd[d]`` an exact Python int for d = -1 .. m-1.
    """
    from itertools import combinations
    S, T = n, n + 1
    naug = n + 2
    m = L + 2
    adj = [[] for _ in range(naug)]
    seen = set()
    for a, b in edges:
        a, b = int(a), int(b)
        if (a, b) in seen or (b, a) in seen:
            continue
        seen.add((a, b))
        adj[a].append(b)
        adj[b].append(a)
    adj[S].append(s)                       # aux edge (S,s), optional
    adj[s].append(S)
    adj[t].append(T)                       # aux edge (t,T), optional
    adj[T].append(t)
    cap = [2] * naug                       # aux NOT forced: S,T free to be inactive
    if order is None:
        order, _ = _boundary_order(naug, [[(u, 1.0) for u in adj[v]]
                                          for v in range(naug)])
    pos = [0] * naug
    for i, v in enumerate(order):
        pos[v] = i
    for v in range(naug):
        adj[v].sort(key=lambda u: pos[u])

    # state: frontier-degree profile (tuple of (vertex,deg)) -> polynomial
    #        {(edges, #degree-1 vertices): weighted count}.  The microstate weight
    #        2^(#real covered) is folded in via `fac` as each real vertex finalises
    #        at degree >= 1; the degree-1 tally (incl. S,T) fixes the deficit.
    states = {(): {(0, 0): 1}}
    max_states = 1
    for v in order:
        nv = pos[v]
        later = [u for u in adj[v] if pos[u] > nv]
        isreal = v < n
        new = defaultdict(dict)
        for st, poly in states.items():
            std = dict(st)
            dv0 = std.pop(v, 0)
            room = cap[v] - dv0
            min_e = min(e for (e, _) in poly)
            for k in range(0, room + 1):
                if min_e + k > m:             # cheapest surviving edge count exceeds m
                    break
                dvf = dv0 + k                  # v's FINAL degree
                fac = 2 if (isreal and dvf >= 1) else 1     # microstate weight
                is1 = 1 if dvf == 1 else 0                  # degree-1 vertex (incl S,T)
                for chosen in combinations(later, k):
                    add = {}
                    ok = True
                    for u in chosen:
                        if std.get(u, 0) + add.get(u, 0) + 1 > cap[u]:
                            ok = False
                            break
                        add[u] = add.get(u, 0) + 1
                    if not ok:
                        continue
                    nd = dict(std)
                    for u, c in add.items():
                        nd[u] = nd.get(u, 0) + c
                    tgt = new[tuple(sorted(nd.items()))]
                    for (e, d1), cnt in poly.items():
                        e2 = e + k
                        if e2 > m:
                            continue
                        key = (e2, d1 + is1)
                        tgt[key] = tgt.get(key, 0) + cnt * fac
        states = new
        max_states = max(max_states, len(states))

    Nd = defaultdict(int)
    for st, poly in states.items():
        for (e, d1), cnt in poly.items():
            if e == m and d1 % 2 == 0:        # #degree-1 vertices must be even
                Nd[d1 // 2 - 1] += cnt
    stats = dict(m=m, L=L, max_states=max_states,
                 order_max_frontier=_max_frontier(
                     naug, [[(u, 1.0) for u in adj[v]] for v in range(naug)], order),
                 micro_weight="per-config microstate weight 2^(#real vertices covered); "
                              "aux edges NOT forced -- S,T may be inactive")
    return dict(Nd), stats


def brute_degree2_by_deficit(n, edges, s, t, L):
    """O(C(|E|+2, m)) full-space reference for tiny graphs.

    Enumerates every m = L+2 edge, degree-<=2 subgraph of the AUGMENTED graph
    (real edges + the two OPTIONAL zero-weight aux edges (S,s),(t,T) -- aux NOT
    forced), weights each by its microstate multiplicity 2^(#real vertices covered),
    and grades it by deficit d = (#degree-1 vertices)/2 - 1 (degree-1 count includes
    the terminals S,T, so the full S-s-...-t-T path is d=0 and a pure disjoint-cycle
    set is d=-1).  Validates ``count_degree2_by_deficit``.
    """
    from itertools import combinations as _comb
    E = [(int(a), int(b)) for a, b in edges]
    S, T = n, n + 1
    cand = E + [(S, s), (t, T)]           # aux edges are optional candidates
    m = L + 2
    Nd = defaultdict(int)
    for cb in _comb(range(len(cand)), m):
        deg = defaultdict(int)
        ok = True
        for i in cb:
            u, v = cand[i]
            deg[u] += 1
            deg[v] += 1
            if deg[u] > 2 or deg[v] > 2:
                ok = False
                break
        if not ok:
            continue
        deg1 = sum(1 for d in deg.values() if d == 1)
        if deg1 % 2:                          # #degree-1 vertices must be even
            continue
        real_covered = sum(1 for x, d in deg.items() if d >= 1 and x < n)
        Nd[deg1 // 2 - 1] += 1 << real_covered
    return dict(Nd)
