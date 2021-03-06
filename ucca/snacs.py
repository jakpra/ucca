import sys, os
import json

import time

from collections import Counter

import matplotlib.pyplot as plt

from ucca import core as ucore, convert as uconv, layer0 as ul0, layer1 as ul1, visualization as uviz


def get_heads(node):
    if node.is_scene():
        return [node.process or node.state]
    else:
        return node.centers


def get_head_terminals(node):
    result = []
    agenda = [node]
    while agenda:
        current = agenda.pop()
        if current.terminals:
            result.extend(current.terminals)
        else:
            agenda.extend(get_heads(current))
    return sorted(result, key=lambda x: x.position)

def siblings(edge:ucore.Edge, exclude=False):
    return sorted([e for e in edge.parent.outgoing if not (exclude and edge.child.ID == e.child.ID)],
                  key=lambda e: e.child.position if type(e.child) == ul0.Terminal else e.child.start_position)

def find_in_siblings(terminal:ul0.Terminal, edge:ucore.Edge):
    try:
        return None if terminal is None else \
            next(sib for sib in siblings(edge) if terminal in sib.child.get_terminals())
    except StopIteration:
        return None

def get_text(node):
    return ' '.join([t.text for t in sorted(node.get_terminals(), key=lambda x: x.position)])


def heuristic_a(edge:ucore.Edge, *, gov_term=None, obj_term=None, **kwargs):
    '''
    Participant or circumstantial modifier of a scene
    :param edge:
    :param passage:
    :param ss: SNACS scene role supersense
    :param gov_term: terminal unit containing gov
    :return: the edge whose FTag is going to be refined with the SNACS scene role
    '''
    parent = edge.parent
    if edge.tag in (ul1.EdgeTags.Relator, ul1.EdgeTags.Function) and parent.fparent is not None and parent.fparent.is_scene:
        if parent.ftag in (ul1.EdgeTags.Process, ul1.EdgeTags.State):

            obj_edge = find_in_siblings(obj_term, parent._fedge())
            gov_edge = find_in_siblings(gov_term, parent._fedge())

            if gov_edge and gov_edge.tag == ul1.EdgeTags.Quantifier:
                return gov_edge
            else:
                return obj_edge

        else:
            sibs = siblings(edge)
            index_in_parent = sibs.index(edge)
            if (index_in_parent == 0 or index_in_parent == len(sibs) - 1):
                return parent._fedge()

def heuristic_b(edge:ucore.Edge, *, gov_term=None, obj_term=None, lexcat='', **kwargs):
    '''
    Configurative or circumstantial modifier of a non-scene
    :param edge:
    :param passage:
    :param ss: SNACS scene role supersense
    :param gov_term: terminal unit containing gov
    :param obj_term: terminal unit containing obj
    :return: the edge whose FTag is going to be refined with the SNACS scene role
    '''
    if edge.tag in (ul1.EdgeTags.Relator, ul1.EdgeTags.Connector, ul1.EdgeTags.Function):
        parent = edge.parent
        sibs = siblings(edge)
        index_in_parent = sibs.index(edge)
        if (index_in_parent == 0 or index_in_parent == len(sibs) - 1) \
            and (parent.fparent is None or not parent.fparent.is_scene) \
            and parent.ftag != ul1.EdgeTags.Center:
            return parent._fedge()
        else:
            if parent.ftag == ul1.EdgeTags.Center:
                obj_edge = find_in_siblings(obj_term, parent._fedge())
                gov_edge = find_in_siblings(gov_term, parent._fedge())
            else:
                obj_edge = find_in_siblings(obj_term, edge)
                gov_edge = find_in_siblings(gov_term, edge)

            if gov_edge and gov_edge.tag == ul1.EdgeTags.Quantifier:
                return gov_edge
            else:
                return obj_edge

def heuristic_c(edge:ucore.Edge, *, lexcat='', obj_term=None, **kwargs):
    '''
    Predication
    :param edge:
    :param passage:
    :param ss: SNACS scene role supersense
    :param lexcat: SNACS lexcat
    :param obj_term: terminal unit containing obj
    :return: the edge whose FTag is going to be refined with the SNACS scene role
    '''
    if edge.tag in (ul1.EdgeTags.State, ul1.EdgeTags.Process):
        if lexcat == 'PRON.POSS' or not obj_term:
            return edge
        else:
            return find_in_siblings(obj_term, edge)

def heuristic_d(edge:ucore.Edge, *, obj_term=None, **kwargs):
    '''
    Linkage
    :param edge:
    :param passage:
    :param ss: SNACS scene role supersense
    :param obj_term: terminal unit containing obj
    :return: the edge whose FTag is going to be refined with the SNACS scene role
    '''
    if edge.tag == ul1.EdgeTags.Linker:
        return find_in_siblings(obj_term, edge)

def heuristic_e(edge:ucore.Edge, *, lexcat='', **kwargs):
    '''
    Intransitive prepositions, "particles", Possessive pronouns
    :param edge:
    :param passage:
    :param ss: SNACS scene role supersense
    :return: the edge whose FTag is going to be refined with the SNACS scene role
    '''
    if lexcat == 'PRON.POSS' or edge.tag in (ul1.EdgeTags.Adverbial, ul1.EdgeTags.Elaborator, ul1.EdgeTags.Participant, ul1.EdgeTags.Time):
        # if lexcat != 'PRON.POSS':
        return edge

def heuristic_f(edge:ucore.Edge, *, lexcat='', **kwargs):
    '''
    Participant possessive pronouns
    :param edge:
    :param passage:
    :param ss: SNACS scene role supersense
    :param lexcat: SNACS lexcat
    :return: the edge whose FTag is going to be refined with the SNACS scene role
    '''
    if edge.tag == ul1.EdgeTags.Participant:
        if lexcat == 'PRON.POSS':
            return edge

def heuristic_g(edge:ucore.Edge, *, lexcat='', **kwargs):
    '''
    Adnominal infinitival purpose markers ("Inherent Purpose")
    :param edge:
    :param passage:
    :param ss: SNACS scene role supersense
    :param lexcat: SNACS lexcat
    :return: the edge whose FTag is going to be refined with the SNACS scene role
    '''
    if edge.tag == ul1.EdgeTags.Function:
        if lexcat == 'INF.P':
            return edge.parent._fedge()

def heuristic_h(edge:ucore.Edge, ss, lexcat='', **kwargs):
    '''
    Approximator
    :param edge:
    :param passage:
    :param ss: SNACS scene role supersense
    :param lexcat: SNACS lexcat
    :return: the edge whose FTag is going to be refined with the SNACS scene role
    '''
    if ss == 'p.Approximator':
        return edge

def find_refined(term:ul0.Terminal, terminals:dict, local=False, ss=None):

    if ss is None:
        if 'ss' not in term.extra:
            return [], {}
        ss = term.extra['ss']

    successes_for_unit = fails_for_unit = 0
    failed_heuristics = []
    abgh = a = b = g = h = c = d = ef = e = f = g = idiom = 0
    e_scn_mod = f_scn_mod = g_scn_mod = h_scn_mod = idiom_scn_mod = 0
    abgh_fail = c_fail = d_fail = ef_fail = e_fail = f_fail = g_fail = 0
    mwe_una_fail = no_match = 0
    warnings = 0

    lexcat = term.extra.get('lexcat')
    # lexlemma = term.extra['lexlemma']
    toknums = sorted(map(int, str(term.extra.get(('local_' if local else '') + 'toknums', '')).split()))
    # span = f'{toknums[0]}-{toknums[-1]}'
    # rel = term.extra['heuristic_relation']
    gov, govlemma = term.extra.get(('local_' if local else '') + 'gov', -1), term.extra.get('govlemma', None)
    obj, objlemma = term.extra.get(('local_' if local else '') + 'obj', -1), term.extra.get('objlemma', None)
    pp_idiom = lexcat == 'PP'
    # if lexcat == 'PP':
    #     obj, objlemma = None, None
    config = term.extra.get('config')

    # terminals = dict(passage.layer('0').pairs)

    gov_term = terminals.get(gov)
    obj_term = terminals.get(obj)

    try:
        unit_terminals = [terminals[toknum] for toknum in toknums]
    except KeyError:
        # print(toknums, file=sys.stderr)
        # print(terminals, file=sys.stderr)
        unit_terminals = None
#        exit(1)
    preterminals = term.parents
    if len(preterminals) != 1:
        # print(term.text, [str(pt) for pt in preterminals])
        return [], {'multiple_preterminals': 1}

    preterminal = preterminals[0]

    if ss[0] != 'p':
        return [], {'non_semrole': 1}

    failed_heuristics = []

    # check whether SNACS mwe is UNA unit in UCCA
    if len(toknums) > 1 and unit_terminals:
        if not all(t.parents[0] == preterminal for t in unit_terminals[1:]):
            # skip SNACS unit if not all tokens are included in UCCA unit
            # fail(unit, None, f'terminals comprising strong MWE are not unanalyzable: [{lexlemma}] in {passage}')
            mwe_una_fail += 1
            warnings += 1
    #        fails_for_unit += 1
            if pp_idiom:
                failed_heuristics.append('PP_idiom_not_UNA')
            else:
                failed_heuristics.append('MWP_not_UNA')
    #         return [], {} #'failed_heuristics':failed_heuristics}

    if len(preterminal.terminals) > len(toknums):
        # warn if UCCA UNA unit is larger than SNACS unit
        # warn(unit, None, f'PSS-bearing token(s) are part of a larger unanalyzable unit: [{lexlemma}] in {passage}')
        failed_heuristics.append('larger_UNA_warn')
        warnings += 1

    # assert len(preterminal.incoming) == 1, str(preterminal) + ' in ' + str(passage)
    refined = []

    for edge in preterminal.incoming:
        if edge is None:
            fails_for_unit += 1
            continue

        ref = None

        if edge.tag == ul1.EdgeTags.Center:
            if edge.parent._fedge() is not None:
               edge = edge.parent._fedge()

        # if edge.attrib.get('remote') or edge.child.attrib.get('implicit'):
        #     refined = None
        #     failed_heuristics.append('REM_IMP')

        if pp_idiom:
            ref = edge
            idiom += 1
            successes_for_unit += 1

        elif lexcat == 'PRON.POSS' or edge.tag in (ul1.EdgeTags.Adverbial, ul1.EdgeTags.Elaborator,
                          ul1.EdgeTags.Participant, ul1.EdgeTags.Time):
            ref = heuristic_e(edge, lexcat=lexcat)

            ef += 1
            if ref:
                if lexcat == 'PRON.POSS':
                    f += 1
                    if ref.parent.is_scene():
                        f_scn_mod += 1
                else:
                    e += 1
                    if ref.parent.is_scene():
                        e_scn_mod += 1
            if ref:
                successes_for_unit += 1
            else:
                ef_fail += 1
                failed_heuristics.append('E')
                fails_for_unit += 1

        elif edge.tag in (ul1.EdgeTags.Relator, ul1.EdgeTags.Connector, ul1.EdgeTags.Function):
            ref = heuristic_h(edge, ss, lexcat=lexcat)
            if ref:
                h += 1
                if ref.parent.is_scene():
                    h_scn_mod += 1
            else:
                ref = heuristic_g(edge, lexcat=lexcat)
                if ref:
                    g += 1
                    if ref.parent.is_scene():
                        g_scn_mod += 1
                else:
                    ref = heuristic_a(edge, gov_term=gov_term, obj_term=obj_term)
                    if ref:
                        a += 1
                    else:
                        ref = heuristic_b(edge, gov_term=gov_term, obj_term=obj_term, lexcat=lexcat)
                        if ref:
                            b += 1

            abgh += 1
            if ref:
                successes_for_unit += 1
            else:
                abgh_fail += 1
                failed_heuristics.append('ABGH')
                fails_for_unit += 1

        elif edge.tag in (ul1.EdgeTags.State, ul1.EdgeTags.Process):
            ref = heuristic_c(edge, lexcat=lexcat, obj_term=obj_term)

            c += 1
            if ref:
                successes_for_unit += 1
            else:
                # print(term.extra)
                # input()
                c_fail += 1
                failed_heuristics.append('C')
                fails_for_unit += 1

        elif edge.tag == ul1.EdgeTags.Linker:
            ref = heuristic_d(edge, obj_term=obj_term)

            d += 1
            if ref:
                successes_for_unit += 1
            else:
                d_fail += 1
                failed_heuristics.append('D')
                fails_for_unit += 1

        # elif edge.tag == ul1.EdgeTags.Participant:
        #     refined = heuristic_f(edge, passage, ss, lexcat=lexcat)
        #
        #     f += 1
        #     if not refined:
        #         f_fail += 1
        #         failed_heuristics.append('F')

        # elif edge.tag == ul1.EdgeTags.Function:
        #     refined = heuristic_g(edge, passage, ss, lexcat=lexcat)
        #
        #     g += 1
        #     if not refined:
        #         g_fail += 1
        #         failed_heuristics.append('G')

        else:
            no_match += 1
            failed_heuristics.append('__ALL__')
            fails_for_unit += 1

        if ref is not None:
            refined.append(ref)

    synt_sem_obj_match = len([e for e in refined if (obj_term is None and e.child == preterminal) or (obj_term is not None and obj_term in e.child.get_terminals())])

    error = {'abgh': abgh, 'a': a, 'b': b, 'c': c, 'd': d, 'ef': ef, 'e': e, 'f': f, 'g': g, 'h': h, 'idiom': idiom,
             'e_scn_mod': e_scn_mod, 'f_scn_mod': f_scn_mod, 'g_scn_mod': g_scn_mod, 'h_scn_mod': h_scn_mod,
             'abgh_fail':abgh_fail, 'c_fail':c_fail, 'd_fail':d_fail, 'e_fail':e_fail, 'ef_fail': ef_fail,
             'no_match':no_match, 'mwe_una_fail':mwe_una_fail,
             'successes_for_unit':successes_for_unit, 'fails_for_unit':fails_for_unit,
             'synt_sem_obj_match': synt_sem_obj_match,
             'warnings':warnings,
             'failed_heuristics':failed_heuristics,
             'remotes': len([e for e in refined if e.attrib.get('remote')])}

    return (refined if len(refined) >= 1 else preterminal.incoming), error

def get_streusle_docs(streusle_file):

    with open(streusle_file) as f:
        streusle = json.load(f)

    docs = {}
    exprs = {}
    _doc_id = None

    unit_counter = 0
    sents = []

    for sent in streusle:
        doc_id, sent_offs = sent['sent_id'].split('-')[-2:]

        if doc_id != _doc_id:
            tok_offs = 0
            if sents:
                # print(_doc_id)
                docs[_doc_id] = {'id': _doc_id, 'sents': sents, 'exprs': exprs, 'toks': toks, 'ends': ends}
            _doc_id = doc_id
            exprs = {}
            toks = []
            sents = []
            ends = []

        sents.append(sent)
        toks.extend(sent['toks'])
        ends.append(len(toks))

        # print('\n', sent_offs, file=sys.stderr)
        for expr in list(sent['swes'].values()) + list(sent['smwes'].values()):
            # print('\t', expr, file=sys.stderr)
            if expr['ss'] and 'heuristic_relation' in expr:
                # print('ok')
                unit_counter += 1
                expr['sent_offs'] = sent_offs
                expr['doc_id'] = doc_id
                expr['local_toknums'] = expr['toknums']
                expr['toknums'] = [t + tok_offs for t in expr['toknums']]
                if expr['heuristic_relation']['gov']:
                    expr['heuristic_relation']['local_gov'], expr['heuristic_relation']['gov'] = \
                        expr['heuristic_relation']['gov'], expr['heuristic_relation']['gov']+tok_offs
                if expr['heuristic_relation']['obj']:
                    expr['heuristic_relation']['local_obj'], expr['heuristic_relation']['obj'] = \
                        expr['heuristic_relation']['obj'], expr['heuristic_relation']['obj'] + tok_offs
                exprs[tuple(expr['toknums'])] = expr

        tok_offs += len(sent['toks'])

    if sents:
        # print(_doc_id)
        docs[_doc_id] = {'id': _doc_id, 'sents': sents, 'exprs': exprs, 'toks': toks, 'ends': ends}

    # print(unit_counter)
    return docs

def get_passages(streusle_file, ucca_path, annotate=True, target='prep', docids=None, ignore=None, diverging_tok=False, token_map={}):


    unit_counter = 0

    for doc_id, doc in get_streusle_docs(streusle_file).items():
        ucca_file = ucca_path + '/xml/' + doc_id + '.xml'
        if (docids and doc_id not in docids):
            print(f'{doc_id} not reviewed')
            continue
        if (ignore and doc_id in ignore):
            print(f'{doc_id} ignored due to diverging tokenization')
            continue
        if not os.path.exists(ucca_file):
            print(f'{ucca_file}: file does not exist')
            continue

        passage = uconv.file2passage(ucca_file)

        tokens = [tok['word'] for tok in doc['toks']]
        terminals = passage.layer('0').pairs

        term2tok = {}
        tok2term = {}

        if diverging_tok:
            j = 0
            acc = ''
            for i, (_, t) in enumerate(terminals):
                term2tok[i] = j
                tok2term[j] = i
                if j >= len(tokens):
                    assert False, (t.text, i, j, len(tokens), tokens)
                tok = tokens[j]
                mapped = token_map.get(tok, tok)
                if mapped.startswith(t.text):
                    acc = t.text
                elif t.text in mapped:
                     acc += t.text
                else:
                    acc = t.text
                if acc == mapped:
                    j += 1
                assert acc in mapped, (acc, mapped)
        else:
            diff_term_tok = len(terminals) - len(tokens)
            if diff_term_tok != 0:
                for (_, term), tok in zip(terminals, tokens):
                    assert tok == term.text
            if diff_term_tok > 0:
                term2tok = tok2term = dict(enumerate(range(len(tokens))))
            else:
                term2tok = tok2term = dict(enumerate(range(len(terminals))))
            #assert len(terminals) == len(
            #        tokens), f'unequal number of UCCA terminals and SNACS tokens: {[t.text for _, t in terminals]}, {tokens}'


        # for x, y in sorted(term2tok.items()):
        #     print(terminals[x][1].text, tokens[y])

        # assert len(terminals) == len(
        #     tokens), f'unequal number of UCCA terminals and SNACS tokens: {[t.text for _, t in terminals]}, {tokens}'

        doc['ends'] = [tok2term[e-1]+1 for e in doc['ends']]

        if annotate:
            for i, (_, term) in enumerate(terminals):
                if i not in term2tok: continue
                tok = doc['toks'][term2tok[i]]
                for k, v in tok.items():
                    if k == 'head' and int(v) > 0:
                        term.extra[k] = str(tok2term[int(v)-1]+1)
                    elif k != '#':
                        term.extra[k] = v

            for unit in list(doc['exprs'].values()):

                unit_counter += 1

                terminal = terminals[tok2term[unit['toknums'][0]-1]][1]
                terminal.extra['toknums'] = ' '.join(map(str, unit['toknums']))
                terminal.extra['local_toknums'] = ' '.join(map(str, unit['local_toknums']))
                terminal.extra['lexlemma'] = unit['lexlemma']
                terminal.extra['lexcat'] = unit['lexcat']
                if unit['lexcat'] == 'DISC':
                    unit['ss'] == '`d'
                terminal.extra['config'] = unit['heuristic_relation']['config']
                terminal.extra.update(unit['heuristic_relation'])
                terminal.extra['gov'] = None if terminal.extra['gov'] is None else tok2term[int(terminal.extra['gov']) - 1] + 1
                terminal.extra['obj'] = None if terminal.extra['obj'] is None else tok2term[int(terminal.extra['obj']) - 1] + 1
                if target == 'obj' and unit['heuristic_relation']['obj'] is not None:
                    obj = terminals[unit['heuristic_relation']['obj']-1][1]
                    obj.extra['ss'] = unit.get('ss', '')
                    obj.extra['ss2'] = unit.get('ss2', '')
                else:
                    terminal.extra['ss'] = unit.get('ss', '')
                    terminal.extra['ss2'] = unit.get('ss2', '')

                # if unit.get('ss', '')[0] == 'p':
                #     unit_counter += 1

        yield doc, passage, term2tok

    # print(unit_counter)


if __name__ == '__main__':

    # ps = list(get_passages('..\\UCCA-SNACS\\data\\the_little_prince\\de\\pss\\lpp_annotation-chpt1-4.govobj.json', '..\\UCCA_German-LPP', annotate=True, target='prep'))

    ignore = """020851
                    020992
                    059005
                    059416
                    200957
                    210066
                    211797
                    216456
                    217359
                    360937
                    399348""".split()

    token_map = {'``':'"', "''":'"', '--':'-'}

    # for doc, passage, term2tok in get_passages('..\\UCCA-SNACS\\data\\the_little_prince\\de\\pss\\lpp_annotation-chpt1-4.govobj.json', '..\\UCCA_German-LPP', annotate=True, target='prep'):
    for doc, passage, term2tok in get_passages(
                '..\\streusle\\train\\streusle.ud_train.govobj.json',
                '..\\UCCA_English-EWT', annotate=True, target='prep', ignore=ignore):


        for pos, terminal in passage.layer('0').pairs:

            if 'ss' not in terminal.extra or (terminal.extra['ss'][0] != 'p' and terminal.extra['ss'] != '`d'):
                # print(terminal.extra)
                continue

            # print('ok')

            start_time = time.time()
            refined, error = find_refined(terminal, dict(passage.layer(ul0.LAYER_ID).pairs))
