"""Joint association of observed silhouettes, without CAD or size priors.

Ordering is a soft cue only: distinct pipes at different depths can exchange
image order. An accepted joint match also needs an independent unary-score
margin; ordering alone never decides between repeated, identical outlines.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


POLICY = {'revision': 'global-silhouette-assignment-v1',
          'maximum_search_nodes': 50000, 'minimum_score_margin_per_changed_pair': .08,
          'minimum_unary_margin_per_changed_pair': .04,
          'axial_support_weight': .1, 'order_inversion_weight': .02,
          'maximum_order_penalty': .04, 'unmatched_observation_cost': .125}
POLICY_SHA256 = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()


def global_assignment(comparisons, evidence, order_cue):
    """Return established pairs plus separated best joint assignments.

    Search each connected compatibility component independently. A missing
    observation may remain unmatched, at an explicit cost. Assignments with
    fewer correspondences also compete, so a good single match beats two
    weak matches. Alternatives must lose by the documented margin for *each*
    promoted edge. A bounded/incomplete search cannot promote any edge.
    ``order_cue`` is diagnostic/soft and must return zero without observed
    common support and non-overlapping silhouettes in both eyes.
    """
    comparisons = np.asarray(comparisons, bool)
    edges = [(int(i), int(j)) for i, j in zip(*np.nonzero(comparisons))]
    local = {pair for pair in edges if comparisons[pair[0]].sum() == 1
             and comparisons[:, pair[1]].sum() == 1}
    accepted = set(local)
    audit = {'revision': POLICY['revision'], 'policy': dict(POLICY),
             'policy_sha256': POLICY_SHA256, 'nominal_dimensions_used': False,
             'ordering_is_hard_constraint': False, 'unmatched_allowed': True,
             'score_definition': 'RELATIVE_IMAGE_MISMATCH_PLUS_UNMATCHED_COST_NOT_PROBABILITY',
             'components': [], 'selected_pairs': []}
    remaining = set(edges)
    while remaining:
        seed = min(remaining)
        lefts, rights = {seed[0]}, {seed[1]}
        while True:
            component = {p for p in remaining if p[0] in lefts or p[1] in rights}
            nl, nr = {p[0] for p in component}, {p[1] for p in component}
            if nl == lefts and nr == rights:
                break
            lefts, rights = nl, nr
        remaining -= component
        row = {'left_indexes': sorted(lefts), 'right_indexes': sorted(rights),
               'pair_evidence': [dict(left_index=i, right_index=j, **evidence[i, j])
                                 for i, j in sorted(component)]}
        if component <= local:
            row.update(status='MUTUAL_UNIQUE', searched_nodes=0)
            audit['components'].append(row)
            continue
        # Search is bounded; incomplete components retain only prior unique
        # matches and never use a partially explored best assignment.
        ordered = sorted(lefts, key=lambda i: (-int(comparisons[i].sum()), i))
        nodes, truncated, hypotheses, order_cache = 0, False, [], {}

        def inversion(a, b):
            key = tuple(sorted((a, b)))
            if key not in order_cache:
                order_cache[key] = order_cue(*key)
            return order_cache[key]

        def visit(index, used, pairs):
            nonlocal nodes, truncated
            nodes += 1
            if nodes > POLICY['maximum_search_nodes']:
                truncated = True
                return
            if index == len(ordered):
                unmatched = len(lefts)+len(rights)-2*len(pairs)
                unary = sum(evidence[p]['score'] for p in pairs)+unmatched*POLICY['unmatched_observation_cost']
                inversions = sum(inversion(a, b) for n, a in enumerate(pairs) for b in pairs[n+1:])
                penalty = min(POLICY['maximum_order_penalty'], inversions * POLICY['order_inversion_weight'])
                hypotheses.append((unary+penalty, unary, frozenset(pairs), inversions))
                return
            i = ordered[index]
            for j in sorted(rights-used):
                if (i, j) in component:
                    visit(index+1, used | {j}, pairs+[(i, j)])
                    if truncated:
                        return
            visit(index+1, used, pairs)

        visit(0, set(), [])
        row.update(searched_nodes=nodes, search_truncated=truncated,
                   assignment_count=len(hypotheses))
        if truncated:
            row['status'] = 'SEARCH_INCOMPLETE'
            audit['components'].append(row)
            continue
        hypotheses.sort(key=lambda h: (h[0], sorted(h[2])))
        best = hypotheses[0]
        row.update(best_score=best[0], best_unary_score=best[1],
                   best_pairs=[list(p) for p in sorted(best[2])],
                   best_order_inversions=best[3], edge_decisions=[])
        promoted = []
        for pair in sorted(best[2]):
            alternatives = [h for h in hypotheses if pair not in h[2]]
            # Normalize by the number of changed correspondences, rather
            # than unrelated unique pipes elsewhere in the same component.
            margins = [(h[0]-best[0])/max(1, len(best[2]-h[2])) for h in alternatives]
            unary_margins = [(h[1]-best[1])/max(1, len(best[2]-h[2])) for h in alternatives]
            separated = not alternatives or (
                min(margins) >= POLICY['minimum_score_margin_per_changed_pair']
                and min(unary_margins) >= POLICY['minimum_unary_margin_per_changed_pair'])
            row['edge_decisions'].append({'left_index': pair[0], 'right_index': pair[1],
                'accepted': separated, 'score_margin': min(margins) if margins else None,
                'unary_margin': min(unary_margins) if unary_margins else None,
                'alternative_assignment_count': len(alternatives)})
            if separated:
                promoted.append(pair)
        accepted.update(promoted)
        row['status'] = 'GLOBAL_SEPARATED' if promoted else 'AMBIGUOUS_GLOBAL_ASSIGNMENTS'
        audit['components'].append(row)
    audit['selected_pairs'] = [{'left_index': i, 'right_index': j,
        'basis': 'MUTUAL_UNIQUE' if (i, j) in local else 'GLOBAL_ONE_TO_ONE'} for i, j in sorted(accepted)]
    return sorted(accepted), audit
