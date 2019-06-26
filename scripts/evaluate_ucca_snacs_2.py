import sys
import glob

from ucca import convert, layer0, layer1, evaluation, normalization
from ucca.constructions import Construction, extract_candidates
from ucca.snacs import find_refined


path = sys.argv[1].replace('\\', '/')
ref_path = sys.argv[2].replace('\\', '/')
#ref_snacs_path = sys.argv[2].replace('\\', '/')
mode = sys.argv[3] # {refinement, concat, pipeline, mtl}
ext_snacs = 'snacs' in sys.argv
if ext_snacs or mode == 'mtl':
    snacs_path = sys.argv[4].replace('\\', '/')
#    ref_snacs_path = sys.argv[5].replace('\\', '/')
integrated = 'integrated' in sys.argv
# preterminal = 'preterminal' in sys.argv
oracle_snacs = 'oracle' in sys.argv

print(path)
print(ref_path)
#print(ref_snacs_path)

def annotate_integrated(term, ss, passage):
    if ss is None:
        return
    refined, error = find_refined(term, dict(passage.layer(layer0.LAYER_ID).pairs), local=True, ss=ss)
    if not refined:
#        assert term.incoming, (term, passage)
        print('something\'s wrong:', term, ss, error)
        if mode == 'concat':
            try:
                refined = term.incoming[0].parent.incoming[0].parent.incoming
            except ValueError:
                refined = []
        else:
            refined = term.incoming[0].parent.incoming if term.incoming else []
    for r in refined:
        # TODO: deal with doubly refined edges
        # if ss not in r.tags:
        r.add(ss)
#        old_tag, old_tags, r.categories = r.tag, r.tags, []
#        all_old_tags = []
#        for t in old_tags:
#            all_old_tags.extend(t.split(':'))
#        r.add(f'{":".join(sorted(all_old_tags))}:{ss}')
#        for tag in old_tags:
#            if tag != old_tag:
#                r.add(f'{tag}:{ss}')

def convert_refinement(passage):
    for edge in (c.edge for _c in extract_candidates(passage).values() for c in _c):
        ss = edge.refinement
        if ss is None: continue
        if ss.startswith('p.'):
            edge.add(ss)
 #       old_tag, old_tags, edge.categories = edge.tag, edge.tags, []
 #       all_old_tags = []
 #       for t in old_tags:
 #           all_old_tags.extend(t.split(':'))
 #       edge.add(f'{":".join(sorted(all_old_tags))}:{ss}')
#        edge.add(f'{old_tag}:{ss}')
#        for tag in old_tags:
#            if tag != old_tag:
#                edge.add(f'{tag}:{ss}')
            #edge.refinement = tag

def convert_concat(passage):
    for edge in (c.edge for _c in extract_candidates(passage).values() for c in _c):
        old_tags, edge.categories = edge.tags, []
        for tag in old_tags:
            ucca_snacs = tag.split(':')
            for t in ucca_snacs:
                if t[0] not in '?`':
                    edge.add(t)

def get_vanilla_ucca(passage):
    _p = convert.join_passages([passage])
    for edge in (c.edge for _c in extract_candidates(_p).values() for c in _c):
        old_tags, edge.categories = edge.tags, []
        for tag in old_tags:
            ucca_snacs = tag.split(':')
            for t in ucca_snacs:
                if t[0] not in 'p?`':
                    edge.add(t)
            #if len(ucca_snacs) >= 2:
            #    edge.refinement = ucca_snacs[1]
    return _p

def get_snacs_refined_ucca(passage):
    p_snacs = convert.join_passages([passage])
    p_refined = convert.join_passages([passage])
    edges_snacs = (c.edge for _c in extract_candidates(p_snacs).values() for c in _c)
    edges_refined = (c.edge for _c in extract_candidates(p_refined).values() for c in _c)
    for e_snacs, e_refined in zip(edges_snacs, edges_refined):
        assert e_snacs.parent.ID == e_refined.parent.ID and e_snacs.child.ID == e_refined.child.ID
        old_tags, e_snacs.categories, e_refined.categories = e_snacs.tags, [], []
        all_old_tags = []
        for tag in old_tags:
            all_old_tags.extend(tag.split(':'))
#        new_tags = []
        if any(t.startswith('p.') for t in all_old_tags):
            for t in all_old_tags:
                if t.startswith('p.'):
                    e_snacs.add(t)
                elif t[0] not in '?`':
                    e_refined.add(t)
                else:
                    assert False, (t, str(e_snacs.parent), str(e_snacs.child), all_old_tags)
#                    if edge not in edges: edges.add((edge, tuple(sorted(all_old_tags))))
#                    edge.add(t)
#        if new_tags:
#            edge.add(':'.join(sorted(new_tags)))
    return p_snacs, p_refined

def get_snacs_ucca(passage):
    _p = convert.join_passages([passage])
    edges = set()
    for edge in (c.edge for _c in extract_candidates(_p).values() for c in _c):
        old_tags, edge.categories = edge.tags, []
        all_old_tags = []
        for tag in old_tags:
            all_old_tags.extend(tag.split(':'))
        if any(t.startswith('p.') for t in all_old_tags):
            for t in all_old_tags:
                if t.startswith('p.'):
                    if edge not in edges: edges.add((edge, tuple(sorted(all_old_tags))))
                    edge.add(t)
    return _p, edges

def get_full_ucca(passage):
    _p = convert.join_passages([passage])
    for edge in (c.edge for _c in extract_candidates(_p).values() for c in _c):
        old_tags, edge.categories = edge.tags, []
        all_old_tags, _ucca, _snacs = [], [], []
        for tag in old_tags:
            for t in tag.split(':'):
                all_old_tags.append(t)
                if t.startswith('p.'):
                    _snacs.append(t)
                else:
                    _ucca.append(t)
#        for t in sorted(_ucca):
#            edge.add(f'{t}:{":".join(sorted(_snacs))}')
        edge.add(f'{":".join(sorted(set(all_old_tags)))}')
#        for tag in old_tags:
#            ucca_snacs = tag.split(':')
#            _tag = ucca_snacs[0]
#            if len(ucca_snacs) >= 2:
#                for t in sorted(ucca_snacs[1:]):
#                    if t.startswith('p.'):
#                        _tag += ':' + t
#            edge.add(_tag)
    return _p

def remove_preterminals(passage):
    _p = convert.join_passages([passage])
    for edge in (c.edge for _c in extract_candidates(_p, constructions=('preterminals',)).values() for c in _c):
#        old_term_edge = terminal.incoming[0]
        non_preterminal_cats, pss = [], []
        for c in edge.categories:
            if c.tag.startswith('Preterminal'):
                tags = c.tag.split(':')
                for t in tags:
                    if t.startswith('p.'):
                        pss.append(t)
            else:
                non_preterminal_cats.append(c.tag)
        assert len(pss) <= 1, (str(edge.parent), pss)
            #if len(tags) >= 2:
            #    refinements += ':' if refinements else '' + ':'.join([t for t in tags[1:] if t.startswith('p.')])
        prepreterminal = edge.parent
        outgoing = [(e.categories, e.child) for e in edge.child.outgoing if isinstance(e.child, layer0.Terminal)]
        assert len(outgoing) <= 1, (prepreterminal, [([c.tag for c in _cats], str(n)) for _cats, n in outgoing])
        if non_preterminal_cats:
            edge.categories = [c for c in edge.categories if not c.tag.startswith('Preterminal')]
            print('WARNING: preterminals and non-preterminals', prepreterminal, outgoing)
        else:
            edge.child.destroy()
            for _cats, n in outgoing:
                new_edge = prepreterminal.add_multiple([(c.tag, '', c.layer, '') for c in _cats] + [(t,) for t in pss] , n)
                if pss:
                    assert n.text
                    new_edge.refinement = pss[0]

    return _p

def extract_mtl_snacs(_glob):
    mtl_preds = {}
    for snacsname in glob.glob(_glob):
        toknums = tuple(int(n) for n in snacsname.split('_')[-1].rsplit('.', maxsplit=1)[0].split('-'))
#        print('\t', toknums, file=sys.stderr)
        snacs_p = convert.xml2passage(snacsname)
        assert snacs_p is not None
        preds_for_unit = []
        target = None
        for (pos, term) in snacs_p.layer(layer0.LAYER_ID).pairs:
            if term is None: continue
            if term.extra.get('identified_for_pss') and target is None:
                target = term
            if term.text == '*ss*':
                pred = None
                node = term
                while node.incoming:
                    edge = node.incoming[0]
                    if edge.tag.startswith('p.'):
                        pred = edge.tag
                        break
                    node = edge.parent
                if pred is not None:
                    preds_for_unit.append(pred)
        assert len(preds_for_unit) <= 1
        if preds_for_unit:
            snacs_info = target.extra
            snacs_info['ss'] = preds_for_unit[0]
            mtl_preds[toknums[0]] = snacs_info

    return mtl_preds


mutual = 0
#only_pred = 0
#only_gold = 0
preds = 0
golds = 0

gold_pred = 0
mutual_mwe = 0

integrated_results = []
vanilla_results = []
snacs_results = []
refined_results = []

with open('edges_refined.tsv', 'w') as f:
    pass
with open('edges_snacs.tsv', 'w') as f:
    pass

for iSent, filename in enumerate(sorted(glob.glob(f'{path}/*.xml'))):
    name = filename.replace('\\', '/').rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=1)[0]
    passage = convert.xml2passage(filename)
    ref = convert.xml2passage(f'{ref_path}/{name}.xml')
#    ref = convert.xml2passage(f'{ref_snacs_path}/{name}.xml')


    if mode == 'mtl':
        # 025516_0002_5-6
        # 025516001
 #       print(name, file=sys.stderr)
        mtl_preds = extract_mtl_snacs(f'{snacs_path}/{name[:-3]}_{int(name[-3:])+1:04d}_*.xml')
        #mtl_golds = extract_mtl_snacs(f'{ref_snacs_path}/{name[:-3]}_{int(name[-3:])+1:04d}_*.snacs')

    elif mode == 'concat' and not integrated:
#        print(passage)
        passage = remove_preterminals(passage)
        # print(passage)

    if integrated:
#        if mode == 'refinement':
#            convert_refinement(passage)
#        convert_refinement_to_concat(ref)
        if mode == 'concat':
            convert_concat(passage)
        if oracle_snacs or ext_snacs:
            passage = get_vanilla_ucca(passage)
            integrated = False

    if ext_snacs:
        snacs = convert.xml2passage(f'{snacs_path}/{name}.xml')
    else:
        snacs = passage


    for (pos, term), (snacs_pos, snacs_term), (ref_pos, ref_term) in zip(passage.layer(layer0.LAYER_ID).pairs, snacs.layer(layer0.LAYER_ID).pairs, ref.layer(layer0.LAYER_ID).pairs):

#            if iSent == 152 and pos == 12:
#                print(passage)
#                print(term)
#                print(name)


#        ref_preterminal = ref_term.incoming[0].parent if ref_term.incoming else None
#        ref_ucca_snacs = ref_preterminal.ftag.split(':') if ref_preterminal and ref_preterminal.incoming else None
#        if ref_ucca_snacs:
#           ref_term.incoming[0].tag = layer1.EdgeTags.Terminal
#            for inc in ref_preterminal.incoming:
#                tags, inc.categories = inc.tags, []
#                for t in tags:
#                    inc.add(t.split(':')[0])
#        gold = ref_ucca_snacs[1] if ref_ucca_snacs is not None and len(ref_ucca_snacs) == 2 else 'NONE'
        gold = ref_term.extra.get('ss')
        if gold is not None and gold.startswith('p.'):
            ref_term.extra['has_gold_ss'] = True
            term.extra['has_gold_ss'] = True

            gold = gold.replace('/', '')
            annotate_integrated(ref_term, gold, ref)

        if gold == '??':
            gold = None
            pred = None

        elif oracle_snacs:
            pred = gold

        elif not integrated:
            if mode == 'mtl':
                if pos in mtl_preds:
                    term.extra.update(mtl_preds[pos])
#                    print(term, mtl_preds[pos])
                #if ref_pos in mtl_golds:
                #    ref_term.extra.update(mtl_golds[ref_pos])
#                    print(ref_term, mtl_golds[ref_pos], file=sys.stderr)

            if mode in ('concat', 'refinement'):
                # refinement
                pred = snacs_term.incoming[0].refinement if snacs_term.incoming else None
                #gold = ref_term.incoming[0].refinement if ref_term.incoming else 'NONE'

#            elif mode == 'concat':
#                # concat
#                term = term.incoming[0].parent if term.incoming else None
#                ucca_snacs = term.ftag.split(':') if term and term.incoming else None
#                ucca_snacs = term.incoming[0].tag.split(':') if term.incoming else None
#                if ucca_snacs:
#                    term.incoming[0].tag = layer1.EdgeTags.Terminal
#                    for inc in term.incoming:
#                        tags, inc.categories = inc.tags, []
#                        for t in tags:
#                            inc.add(t.split(':')[0])
#                pred = ucca_snacs[1] if ucca_snacs is not None and len(ucca_snacs) == 2 else 'NONE'

            elif mode in ('pipeline', 'mtl'):
                pred = snacs_term.extra.get('ss')
#                gold = ref_term.extra.get('ss', 'NONE').replace("/", "")


        if oracle_snacs or not integrated:

            if gold == '??':
                gold = None
                pred = None
            else:
                pred = pred.replace('/', '') if pred and pred.startswith('p.') else None
                gold = gold.replace('/', '') if gold and gold.startswith('p.') else None


            annotate_integrated(term, pred, passage)

#            if pred or gold:
#                print(term, pred, gold, iSent, pos, sep='\t')

            if pred is not None:
                preds += 1
            if gold is not None:
                golds += 1
                if pred is not None:
                    gold_pred += 1
                if pred == gold:
                    mutual += 1
                    if snacs_term.extra.get('toknums') == ref_term.extra.get('toknums'):
                       mutual_mwe += 1
 #               else:
 #                   only_gold += 1
 #                   only_pred += 1 if pred is not None else 0
 #           elif pred is not None:
 #               only_pred += 1

#    normalization.normalize(passage)
#    normalization.normalize(ref)

    print(passage)
    print(ref)
    print()

    passage_full = get_full_ucca(passage)
    passage_vanilla = get_vanilla_ucca(passage)
    passage_snacs, passage_refined = get_snacs_refined_ucca(passage)
#    passage_refined, edges_refined = get_refined_ucca(passage)
    ref_full = get_full_ucca(ref)
    ref_vanilla = get_vanilla_ucca(ref)
    ref_snacs, ref_refined = get_snacs_refined_ucca(ref)
#    ref_refined, edges_refined = get_refined_ucca(ref)

#    with open('edges_snacs.tsv', 'a') as f:
#        for e, ts in sorted(edges_snacs,key=lambda x: str(x[0])):
#            print(name, e, ts, sep='\t', file=f)

#    with open('edges_refined.tsv', 'a') as f:
#        for e, ts in sorted(edges_refined, key=lambda x: str(x[0])):
#            print(name, e, ts, sep='\t', file=f)

#    print(passage_snacs)
#    print(ref_snacs)

    integrated_results.append(evaluation.evaluate(passage_full, ref_full, constructions=('Non-preterm','SNACS'), normalize=False))
    vanilla_results.append(evaluation.evaluate(passage_vanilla, ref_vanilla, constructions=('Non-preterm','SNACS', 'has_gold_SNACS', 'has_gold_SNACS_sibling', 'has_gold_SNACS_or_sibling', 'scenes', 'scene_children', 'scenes_and_scene_children'), normalize=True))
    snacs_results.append(evaluation.evaluate(passage_snacs, ref_snacs, constructions=('Non-preterm', 'SNACS', 'has_tags',), normalize=False))
    refined_results.append(evaluation.evaluate(passage_refined, ref_refined, constructions=('Non-preterm', 'has_tags',), normalize=False))



print('UCCA SNACS')

integ_aggr = evaluation.Scores.aggregate(integrated_results)
integ_aggr.print()
# integ_aggr.print_confusion_matrix()


print('\n\nUCCA')

evaluation.Scores.aggregate(vanilla_results).print()

print('\n\nintegrated SNACS')

evaluation.Scores.aggregate(snacs_results).print()

print('\n\nrefined UCCA')

evaluation.Scores.aggregate(refined_results).print()


print('\n\ntoken-based SNACS')

p_mwe = (mutual_mwe / preds) if mutual_mwe else 1
r_mwe = (mutual_mwe / golds) if mutual_mwe else 0
f_mwe = (2*p_mwe*r_mwe)/(p_mwe+r_mwe)

p = (mutual / preds) if mutual else 1
r = (mutual / golds) if mutual else 0
f = (2*p*r)/(p+r)

print('Precision')
print(f'{100*p:.1f}\t ({mutual}/{preds})')

print('\nRecall')
print(f'{100*r:.1f}\t ({mutual}/{golds})')

print('\nP\tR\tF')
print(f'{100*p:.1f}\t{100*r:.1f}\t{100*f:.1f}')


print('\n\ntoken-based SNACS (MWE strict)')

print('Precision')
print(f'{100*p_mwe:.1f}\t ({mutual_mwe}/{preds})')

print('\nRecall')
print(f'{100*r_mwe:.1f}\t ({mutual_mwe}/{golds})')

print('\nP\tR\tF')
print(f'{100*p_mwe:.1f}\t{100*r_mwe:.1f}\t{100*f_mwe:.1f}')


