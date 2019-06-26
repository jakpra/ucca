import sys
import glob

from ucca import convert, layer0, layer1, evaluation
from ucca.constructions import Construction, extract_candidates
from ucca.snacs import find_refined


path = sys.argv[1].replace('\\', '/')
ref_path = sys.argv[2].replace('\\', '/')
mode = sys.argv[3] # {refinement, concat, pipeline, mtl}
if mode == 'mtl':
    snacs_path = sys.argv[4].replace('\\', '/')
    ref_snacs_path = sys.argv[5].replace('\\', '/')
integrated = 'integrated' in sys.argv
# preterminal = 'preterminal' in sys.argv

print(path)
print(ref_path)

def annotate_integrated(term, ss, passage):
    if ss is None:
        return
    refined, error = find_refined(term, dict(passage.layer(layer0.LAYER_ID).pairs, local=True))
    if not refined:
#        assert term.incoming, (term, passage)
        if mode == 'concat':
            try:
                refined = term.incoming[0].parent.incoming[0].parent.incoming
            except ValueError:
                refined = []
        else:
            refined = term.incoming[0].parent.incoming if term.incoming else []
    for r in refined:
        # TODO: deal with doubly refined edges
        old_tag, old_tags, r.categories = r.tag, r.tags, []
        r.add(f'{old_tag}:{ss}')
        for tag in old_tags:
            if tag != old_tag:
                r.add(f'{tag}')

def convert_refinement_to_concat(passage):
    for edge in (c.edge for _c in extract_candidates(passage).values() for c in _c):
        ss = edge.refinement
        if ss is None: continue
        old_tag, old_tags, edge.categories = edge.tag, edge.tags, []
        edge.add(f'{old_tag}:{ss}')
        for tag in old_tags:
            if tag != old_tag:
                edge.add(f'{tag}')
            #edge.refinement = tag

def get_vanilla_ucca(passage):
    _p = convert.join_passages([passage])
    for edge in (c.edge for _c in extract_candidates(_p).values() for c in _c):
        old_tags, edge.categories = edge.tags, []
        for tag in old_tags:
            ucca_snacs = tag.split(':')
            edge.add(ucca_snacs[0])
            #if len(ucca_snacs) >= 2:
            #    edge.refinement = ucca_snacs[1]
    return _p

def get_refined_ucca(passage):
    _p = convert.join_passages([passage])
    for edge in (c.edge for _c in extract_candidates(_p).values() for c in _c):
        old_tags, edge.categories = edge.tags, []
        for tag in old_tags:
            ucca_snacs = tag.split(':')
            if len(ucca_snacs) >= 2:
                if any(t.startswith('p.') for t in ucca_snacs[1:]):
                    edge.add(ucca_snacs[0])
    return _p

def get_snacs_ucca(passage):
    _p = convert.join_passages([passage])
    for edge in (c.edge for _c in extract_candidates(_p).values() for c in _c):
        old_tags, edge.categories = edge.tags, []
        for tag in old_tags:
            ucca_snacs = tag.split(':')
            if len(ucca_snacs) > 1:
                for t in ucca_snacs[1:]:
                    edge.add(t)
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

mutual_mwe = 0

integrated_results = []
vanilla_results = []
snacs_results = []
refined_results = []

for iSent, filename in enumerate(sorted(glob.glob(f'{path}/*.xml'))):
    name = filename.replace('\\', '/').rsplit('/', maxsplit=1)[-1].rsplit('.', maxsplit=1)[0]
    passage = convert.xml2passage(filename)
    ref = convert.xml2passage(f'{ref_path}/{name}.xml')


    if mode == 'mtl':
        # 025516_0002_5-6
        # 025516001
 #       print(name, file=sys.stderr)
        mtl_preds = extract_mtl_snacs(f'{snacs_path}/{name[:-3]}_{int(name[-3:])+1:04d}_*.xml')
        mtl_golds = extract_mtl_snacs(f'{ref_snacs_path}/{name[:-3]}_{int(name[-3:])+1:04d}_*.snacs')

    if not integrated:
        for (pos, term), (ref_pos, ref_term) in zip(passage.layer(layer0.LAYER_ID).pairs, ref.layer(layer0.LAYER_ID).pairs):

#            if iSent == 152 and pos == 12:
#                print(passage)
#                print(term)
#                print(name)

            if mode == 'mtl':
                if pos in mtl_preds:
                    term.extra.update(mtl_preds[pos])
#                    print(term, mtl_preds[pos])
                if ref_pos in mtl_golds:
                    ref_term.extra.update(mtl_golds[ref_pos])
#                    print(ref_term, mtl_golds[ref_pos], file=sys.stderr)

            if mode == 'refinement':
                # refinement
                pred = term.incoming[0].refinement if term.incoming else 'NONE'
                gold = ref_term.incoming[0].refinement if ref_term.incoming else 'NONE'

            elif mode == 'concat':
                # concat
                preterminal = term.incoming[0].parent if term.incoming else None
                ucca_snacs = preterminal.ftag.split(':') if preterminal and preterminal.incoming else None
#                ucca_snacs = term.incoming[0].tag.split(':') if term.incoming else None
                if ucca_snacs:
#                    term.incoming[0].tag = layer1.EdgeTags.Terminal
                    for inc in preterminal.incoming:
                        tags, inc.categories = inc.tags, []
                        for t in tags:
                            inc.add(t.split(':')[0])
                pred = ucca_snacs[1] if ucca_snacs is not None and len(ucca_snacs) == 2 else 'NONE'
                ref_preterminal = ref_term.incoming[0].parent if ref_term.incoming else None
                ref_ucca_snacs = ref_preterminal.ftag.split(':') if ref_preterminal and ref_preterminal.incoming else None
                if ref_ucca_snacs:
#                    ref_term.incoming[0].tag = layer1.EdgeTags.Terminal
                    for inc in ref_preterminal.incoming:
                        tags, inc.categories = inc.tags, []
                        for t in tags:
                            inc.add(t.split(':')[0])
                gold = ref_ucca_snacs[1] if ref_ucca_snacs is not None and len(ref_ucca_snacs) == 2 else 'NONE'

            elif mode in ('pipeline', 'mtl'):
                pred = term.extra.get('ss', 'NONE').replace("/", "")
                gold = ref_term.extra.get('ss', 'NONE').replace("/", "")

            annotate_integrated(term, pred, passage)
            annotate_integrated(ref_term, gold, ref)

            if gold == '??':
                gold = None
                pred = None
            else:
                pred = pred if pred and pred.startswith('p.') else None
                gold = gold if gold and gold.startswith('p.') else None

            if pred or gold:
                print(term, pred, gold, iSent, pos, sep='\t')

            if pred is not None:
                preds += 1
            if gold is not None:
                golds += 1
                if pred == gold:
                    mutual += 1
                    if term.extra.get('toknums') == ref_term.extra.get('toknums'):
                       mutual_mwe += 1
 #               else:
 #                   only_gold += 1
 #                   only_pred += 1 if pred is not None else 0
 #           elif pred is not None:
 #               only_pred += 1

        print(passage)
        print(ref)

    elif mode == 'refinement':
        convert_refinement_to_concat(passage)
        convert_refinement_to_concat(ref)

    passage_vanilla = get_vanilla_ucca(passage)
    passage_snacs = get_snacs_ucca(passage)
    passage_refined = get_refined_ucca(passage)
    ref_vanilla = get_vanilla_ucca(ref)
    ref_snacs = get_snacs_ucca(ref)
    ref_refined = get_refined_ucca(ref)

#    print(passage_snacs)
#    print(ref_snacs)

    integrated_results.append(evaluation.evaluate(passage, ref, constructions=('Non-preterm',)))
    vanilla_results.append(evaluation.evaluate(passage_vanilla, ref_vanilla, constructions=('Non-preterm',)))
    snacs_results.append(evaluation.evaluate(passage_snacs, ref_snacs, constructions=('Non-preterm', 'SNACS','hastags',)))
    refined_results.append(evaluation.evaluate(passage_refined, ref_refined, constructions=('Non-preterm', 'hastags',)))


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


