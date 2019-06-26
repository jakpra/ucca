import sys, os
import json

import time

from collections import Counter

import matplotlib.pyplot as plt

from ucca import core as ucore, convert as uconv, layer0 as ul0, layer1 as ul1, constructions as uconstr

from ucca.snacs import get_passages, find_refined

def main(args):
    try:
        integrate_full = True
        integrate_term = False
        concatenate = False
        pss_feature = False
        annotate = True
        object = False
        v2_only = True
        draw = False
        output = True
        inp_ucca = False
        if '-I' in args:
            args.remove('-I')
            args.append('--no-integrate')
        if '--no-integrate' in args:
            integrate_full = False
            args.remove('--no-integrate')

        if '-c' in args:
            args.remove('-c')
            args.append('--concatenate')
        if '--concatenate' in args:
            concatenate = True
            args.remove('--concatenate')

        if '-A' in args:
            args.remove('-A')
            args.append('--no-annotate')
        if '--no-annotate' in args:
            integrate_full = False
            annotate = False
            args.remove('--no-annotate')

        if '-s' in args:
            args.remove('-s')
            args.append('--pss-feature')
        if '--pss-feature' in args:
            pss_feature = True
            args.remove('--pss-feature')

        if '--term' in args:
            integrate_term = True
            integrate_full = False
            args.remove('--term')

        if '--inp_ucca' in args:
            inp_ucca = True
            args.remove('--inp_ucca')

        if '-o' in args:
            args.remove('-o')
            args.append('--object')
        if '--object' in args:
            object = True
            args.remove('--object')

        if '-n' in args:
            args.remove('-n')
            args.append('--no-output')
        if '--no-output' in args:
            output = False
            args.remove('--no-output')

        if '--all' in args:
            v2_only = False
            args.remove('--all')

        if '--draw' in args:
            draw = True
            args.remove('--draw')
            import visualization as uviz
            import matplotlib.pyplot as plt

        streusle_file = args[0] #'../../streusle/streusle.govobj.json' #args[0] #'streusle.govobj.json'  # sys.argv[1]
        ucca_path = args[1] #'../../UCCA_English-EWT' #args[1] # '/home/jakob/nert/corpora/UCCA_English-EWT/xml'  # sys.argv[2]
        out_dir = args[2]

    except:
        print(f'usage: python3 {sys.argv[0]} STREUSLE_JSON UCCA_PATH OUT_DIR', file=sys.stderr)
        exit(1)

    with open(streusle_file) as f:
        streusle = json.load(f)

    print()

    global_error = Counter()


    unit_counter = 0
    successful_units = 0
    unsuccessful_units = 0
    deductible_multiple_successes = 0
    deductible_multiple_fails = 0
    deductible_fail_and_success = 0
    units_with_remote = 0

    doc_error = 0

    primary_edges = 0
    remote_edges = 0


    _doc_id = None

    v2_docids = set()
    if v2_only:
        with open(ucca_path + '/v2.txt') as f:
            for line in f:
                v2_docids.add(line.strip())

    ignore = []
    #"""020851
    #            020992
    #            059005
    #            059416
    #            200957
    #            210066
    #            211797
    #            216456
    #            217359
    #            360937
    #            399348""".split()

    unit_times = []

    # print('usnacs.get_passages(streusle_file, ucca_path, annotate=(integrate or annotate), ignore=ignore, docids=v2_docids)')

    tag_refinements = Counter()

    for doc, passage, term2tok in get_passages(streusle_file, ucca_path, annotate=(integrate_term or integrate_full or annotate), target='obj' if object else 'prep', ignore=ignore, docids=v2_docids):

        if output and (not integrate_full and not integrate_term):
            for p in uconv.split_passage(passage, doc['ends'], map(lambda x: ''.join(x['sent_id'].split('-')[-2:]), doc['sents'])):
                uconv.passage2file(p, out_dir + '/' + p.ID + '.xml')
            continue

        l1 = passage.layer('1')

        if not output:
            primary_edges += len(uconstr.extract_candidates(passage, constructions=(uconstr.PRIMARY,))['primary'])
            remote_edges += len(uconstr.extract_candidates(passage, constructions=uconstr.get_by_names(['remote']))['remote'])


        for terminal in passage.layer('0').words:

            if integrate_term and concatenate: # and not terminal.incoming[0].parent.tag.startswith('Preterminal'):
                old_term_edge = terminal.incoming[0]
                preterminal = old_term_edge.parent
                preterminal._outgoing.remove(old_term_edge)
                terminal._incoming.remove(old_term_edge)
                passage._remove_edge(old_term_edge)
#                old_preterm_edge = preterminal._fedge()
#                preterminal.fparent._outgoing.remove(old_preterm_edge)
                new_preterminal = l1.add_fnode(preterminal, 'Preterminal') #[[c.tag, '', c.layer, ''] for c in old_preterm_edge.categories])
#                passage._add_node(new_preterminal)
                #for outg in preterminal.outgoing:
                    #if inc.parent != preterminal.fparent and ul1.EdgeTags.Terminal not in inc.tags:
#                new_preterminal.add(ul1.EdgeTags.Terminal, terminal)
#                passage._add_node(new_preterminal)
                #preterminal._incoming = []
#                new_preterminal.add('Preterminal', preterminal)
#                passage._remove_edge(old_term_edge)
                new_preterminal.add_multiple([[c.tag, '', c.layer, ''] for c in old_term_edge.categories], terminal)
#                assert preterminal.outgoing
#                assert new_preterminal.outgoing
#                print(preterminal)
#                print(new_preterminal)
#                print(terminal)

            pss_label = ''
            if 'ss' in terminal.extra:
                pss_label = terminal.extra['ss']
            if not pss_label.startswith('p'):
                # print(terminal.extra)
                continue

            # print('ok')

            start_time = time.time()
            unit_counter += 1

            if integrate_term:
                if concatenate:
#                    old_term_edge = terminal.incoming[0]
#                    preterminal = old_term_edge.parent
#                    new_preterminal = l1.add_fnode(preterminal, 'Preterminal')
#                    passage._add_node(new_preterminal)
#                    old_term_edge.parent._outgoing.remove(old_term_edge)
#                    old_term_edge.child._incoming.remove(old_term_edge)
#                    passage._remove_edge(old_term_edge)
#                    new_term_edge = new_preterminal.add(ul1.EdgeTags.Terminal, terminal)
#                    passage._add_edge(new_term_edge)
#                    refined = new_preterminal.incoming
                    refined = terminal.incoming[0].parent.incoming
                else:
                    refined = terminal.incoming
            else:
                refined, error = find_refined(terminal, dict(passage.layer(ul0.LAYER_ID).pairs))

                global_error += Counter({k: v for k, v in error.items() if isinstance(v, int)})

                if error['successes_for_unit'] >= 1:
                    successful_units += 1
                    deductible_multiple_successes += error['successes_for_unit'] - 1
                    if error['fails_for_unit'] >= 1:
                        deductible_fail_and_success += 1
                else:
                    unsuccessful_units += 1

                if error['fails_for_unit'] >= 1:
                    deductible_multiple_fails += error['fails_for_unit'] - 1

                if error['remotes'] >= 1:
                    units_with_remote += 1

                if not output:
                    if 'larger_UNA_warn' in error['failed_heuristics']:
                        print(terminal, terminal.incoming[0].parent)

                    if 'PP_idiom_not_UNA' in error['failed_heuristics']:
                        print('PP_idiom:', terminal.extra['lexlemma'], terminal, terminal.incoming[0].parent)

                    if 'MWP_not_UNA' in error['failed_heuristics']:
                        print('MWP:', terminal.extra['lexlemma'], terminal, terminal.incoming[0].parent)


            for r in refined:
                # TODO: deal with doubly refined edges
                if (not concatenate and r.refinement) or (concatenate and ':' in r.tag):
                    pass
                else:
                    if concatenate:
                        cats, r.categories = r.categories, []
                        for c in cats:
                            composit_tag = f'{c.tag}:{pss_label}'
                            r.add(composit_tag)
                            tag_refinements[composit_tag] += 1
                    else:
                        r.refinement = pss_label
#                print('FAIL', doc['id'], terminal.extra['toknums'], terminal.extra['lexlemma'])


            unit_times.append(time.time() - start_time)

            if not pss_feature:
                terminal.extra.pop('ss') # ensuring pss is not also a feature

#            if integrate_term:
#                terminal.extra['identified_for_pss'] = str(True)

        if draw:
            for sent, psg in zip(doc['sents'], uconv.split_passage(passage, doc['ends'])):
                uviz.draw(psg)
                plt.savefig(f'../graphs/{sent["sent_id"]}.svg')
                plt.clf()

#        print(passage)
        if output:
            for p in uconv.split_passage(passage, doc['ends'],
                map(lambda x: ''.join(x['sent_id'].split('-')[-2:]), doc['sents'])):
#                print(p)
#            augmented = uconv.join_passages([p, ucore.Passage('0')])
#            for root_edge in augmented.layer(ul1.LAYER_ID)._head_fnode.outgoing:
#                if len(root_edge.tag.split('-')) > 1:
#                    assert False, augmented
#                root_edge.tag = root_edge.tag.split('-')[0]
                uconv.passage2file(p, out_dir + '/' + p.ID + '.xml')


    for x, y in tag_refinements.most_common(len(tag_refinements)):
        print(x, y, sep='\t')





    #print(f'successful units\t{successful_units}\t{100*successful_units/(unit_counter-doc_error)}%')
    #print(f'unsuccessful units\t{unsuccessful_units}\t{100-(100*successful_units/(unit_counter-doc_error))}%') #={unit_counter - doc_error - successful_units}={mwe_una_fail+abgh_fail+c_fail+d_fail+e_fail+f_fail+g_fail+no_match}

    if integrate_full and not output:

        print('\n\n')
        print(f'total units\t{unit_counter}')
 #   print(f'gov and obj present\t{gov_and_obj_counter}')
        print(f'document error\t{doc_error}\t{100*doc_error/unit_counter}%')
        print(f'document success\t{unit_counter - doc_error}\t{100-(100 * doc_error / unit_counter)}%')
        print(f'total primary edges\t{primary_edges}')
        print(f'total remote edges\t{remote_edges}')
        print('----------------------------------------------------')
        print(f'successful units\t{successful_units}\t{100*successful_units/(unit_counter-doc_error)}%')
        print(f'unsuccessful units\t{unsuccessful_units}\t{100-(100*successful_units/(unit_counter-doc_error))}%') #={unit_counter - doc_error - successful_units}={mwe_una_fail+abgh_fail+c_fail+d_fail+e_fail+f_fail+g_fail+no_match}
        print(f'warnings\t{global_error["warnings"]}')
        print('---------------------------------')
#    for ftype, count in fail_counts.most_common():
#        print(f'{ftype}\t{count}')
        print(f'syntactic and semantic obj match\t{global_error["synt_sem_obj_match"]}')
        print('---------------------------------')
        print(f'\tMWE but not UNA\t{global_error["mwe_una_fail"]}')
        print(f'\tPP idiom\t{global_error["idiom"]}')
        print(f'\tR, N, F ({global_error["abgh"]}) but A and B miss\t{global_error["abgh_fail"]}')
        print(f'\tA (scene mod)\t{global_error["a"]}')
        print(f'\tB (non-scene mod) \t{global_error["b"]}')

        print(f'\tG (inh purpose) \t{global_error["g"]}')
        print(f'\t  scn \t{global_error["g_scn_mod"]}')
        print(f'\t  non scn \t{global_error["g"] - global_error["g_scn_mod"]}')

        print(f'\tH (approximator) \t{global_error["h"]}')
        print(f'\t  scn \t{global_error["h_scn_mod"]}')
        print(f'\t  non scn \t{global_error["h"] - global_error["h_scn_mod"]}')

        print(f'\tP, S ({global_error["c"]}) but C miss\t{global_error["c_fail"]}')
        print(f'\tL ({global_error["d"]}) but D miss\t{global_error["d_fail"]}')

        print(f'\tA, D, E, T ({global_error["ef"]}) but E miss\t{global_error["ef_fail"]}')

        print(f'\tE (intr adp) \t{global_error["e"]}')
        print(f'\t  scn \t{global_error["e_scn_mod"]}')
        print(f'\t  non scn \t{global_error["e"] - global_error["e_scn_mod"]}')

        print(f'\tF (poss pron) \t{global_error["f"]}')
        print(f'\t  scn \t{global_error["f_scn_mod"]}')
        print(f'\t  non scn \t{global_error["f"] - global_error["f_scn_mod"]}')

    #print(f'\tA ({f}) but F miss\t{f_fail}')
    #print(f'\tF ({g}) but G miss\t{g_fail}')
        print(f'\tno match\t{global_error["no_match"]}') #\t{ucca_categories}')
        print(f'\tnon-semantic role\t{global_error["non_semrole"]}')
        print(f'\tmultiple preterminals\t{global_error["multiple_preterminals"]}')
        print(f'\tunits with remote\t{units_with_remote} (total {global_error["remotes"]})')
    #
    #
        print('---------------------------------')
        print(f'\tdeductible (multiple successes for single unit)\t{deductible_multiple_successes}')
        print(f'\tdeductible (multiple fails for single unit)\t{deductible_multiple_fails}')
        print(f'\tdeductible (fail and success for single unit)\t{deductible_fail_and_success}')
    #print(f'config considered {tuple(considered)}: {considered_counter}')
    #print(f'linked as expected: {linked_as_expected_counter}')
    #
    #
    #print(json.dumps(to_SNACS, indent=2))
    #print(json.dumps(to_UCCA, indent=2))




if __name__ == '__main__':
    #main(sys.argv[1:])
    #HOME = 'C:/Users/Jakob/AppData/Local/Packages/CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc/LocalState/rootfs/home/jakob'
    #main([f'{HOME}/nert/streusle/streusle.govobj.json', f'{HOME}/nert/corpora/UCCA_English-EWT'])

    main(sys.argv[1:])
    # args = sys.argv[1:]
    #
    # try:
    #     integrate = True
    #     annotate = True
    #     if '-I' in args:
    #         args.remove('-I')
    #         args.append('--no-integrate')
    #     if '--no-integrate' in args:
    #         integrate = False
    #         args.remove('--no-integrate')
    #
    #     if '-A' in args:
    #         args.remove('-A')
    #         args.append('--no-annotate')
    #     if '--no-annotate' in args:
    #         integrate = False
    #         annotate = False
    #         args.remove('--no-annotate')
    #     streusle_file = args[0] #'../../streusle/streusle.govobj.json' #args[0] #'streusle.govobj.json'  # sys.argv[1]
    #     ucca_path = args[1] #'../../UCCA_English-EWT' #args[1] # '/home/jakob/nert/corpora/UCCA_English-EWT/xml'  # sys.argv[2]
    #     out_dir = args[2]
    #     #start_unit = int(args[-1])
    #     #streusle2ucca_file = '/home/jakob/nert/corpora/UCCA_English-EWT/streusle2ucca.txt'  # sys.argv[3]
    # except:
    #     print(f'usage: python3 {sys.argv[0]} STREUSLE_JSON UCCA_PATH OUT_DIR', file=sys.stderr)
    #     exit(1)
    #
    # with open(streusle_file) as f:
    #     streusle = json.load(f)
    #
    # print()
    #
    # unit_counter = 0
    # successful_units = 0
    # unsuccessful_units = 0
    # warnings = 0
    #
    # doc_error = 0
    #
    # fail_counts = Counter()
    #
    # n_figure = 0
    #
    # docs = {}
    # exprs = {}
    # toks = []
    # sents = []
    # ends = []
    # _doc_id = None
    # l_units = 0
    #
    # v2_docids = set()
    # # with open(ucca_path + '/v2.txt') as f:
    # #     for line in f:
    # #         v2_docids.add(line.strip())
    #
    # ignore = """020851
    #             020992
    #             059005
    #             059416
    #             200957
    #             210066
    #             211797
    #             216456
    #             217359
    #             360937
    #             399348""".split()
    #
    # unit_times = []
    #
    # print('usnacs.get_passages(streusle_file, ucca_path, annotate=(integrate or annotate), ignore=ignore, docids=v2_docids)')
    #
    # tag_refinements = Counter()
    #
    # for doc, passage, term2tok in usnacs.get_passages(streusle_file, ucca_path, annotate=True, target='prep'):  # , ignore=ignore, docids=v2_docids):
    #     if not integrate:
    #         for p in uconv.split_passage(passage, doc['ends'], map(lambda x: ''.join(x['sent_id'].split('-')[-2:]), doc['sents'])):
    #             uconv.passage2file(p, out_dir + '/' + p.ID + '.xml')
    #         continue
    #
    #     # for _, unit in sorted(doc['exprs'].items()):
    #
    #     for pos, terminal in passage.layer('0').pairs:
    #
    #         if 'ss' not in terminal.extra or (terminal.extra['ss'][0] != 'p' and terminal.extra['ss'] != '`d'):
    #             # print(terminal.extra)
    #             continue
    #
    #         # print('ok')
    #
    #         start_time = time.time()
    #         unit_counter += 1
    #
    #         refined, error = usnacs.find_refined(terminal, dict(passage.layer(ul0.LAYER_ID).pairs))
    #
    #         # if doc_id == '231203' and unit['sent_offs'] == '0005':
    #         #     print(unit)
    #         #     print(passage)
    #         #     print(error)
    #
    #         for r in refined:
    #             # TODO: deal with doubly refined edges
    #             if r.refinement:
    #                 pass
    #             else:
    #                 r.refinement = terminal.extra['ss']
    #                 tag_refinements[f'{r.tag}-{r.refinement}'] += 1
    #
    #         # TODO
    #         if len(refined) >= 1:
    #             successful_units += 1
    #
    #             # uviz.draw(passage) #, highlit_ids=[n.ID for n in unit_terminals])
    #             # mng = plt.get_current_fig_manager()
    #             # mng.window.state("zoomed")
    #             # plt.show()
    #
    #             # print(passage)
    #             # if input('[N(ext)/q(uit)]').lower() in ('q', 'quit'):
    #             #     exit(0)
    #
    #         # TODO
    #         else:
    #             unsuccessful_units += 1
    #
    #             print('FAIL', doc['id'], terminal.extra['toknums'], terminal.extra['lexlemma'])
    #
    #             # print(unit)
    #             # print(error)
    #             # print(passage)
    #             # if input('[N(ext)/q(uit)]').lower() in ('q', 'quit'):
    #             #     exit(0)
    #
    #         #     for fail in failed_heuristics:
    #         #         fail_counts[fail] += 1
    #             # print(json.dumps(unit, indent=2))
    #             # # print(passage)
    #             # print(failed_heuristics)
    #             # n_figure += 1
    #             # # if n_figure >= 20:
    #             # #     plt.show()
    #             # #     n_figure = 1
    #             # # plt.figure(n_figure)
    #             # uviz.draw(passage, highlit_ids=[n.ID for n in unit_terminals])
    #             #
    #             # mng = plt.get_current_fig_manager()
    #             # mng.window.state("zoomed")
    #             # plt.show()
    #             # # node_link_data, pos = uviz.draw(passage), uviz.topological_layout(passage)
    #             # #with open('C:/Users/Jakob/Documents/nert/ucca-streusle/graphs/tmp.json', 'w') as f:
    #             # #    json.dump(node_link_data, f, indent=2)
    #             # #with open('C:/Users/Jakob/Documents/nert/ucca-streusle/graphs/tmp_pos.json', 'w') as f:
    #             # #    json.dump(pos, f, indent=2)
    #             # # show(node_link_data, pos)
    #             # # input('\nhit ENTER to continue...')
    #             # continue
    #
    #         # progress = 100 * unit_counter / l_units
    #         # print(f'{unit_counter} ({progress:.2f}%)')
    #
    #         unit_times.append(time.time() - start_time)
    #
    #
    #     for sent, psg in zip(doc['sents'], uconv.split_passage(passage, doc['ends'])):
    #         uviz.draw(psg)
    #         plt.savefig(f'../graphs/{sent["sent_id"]}.svg')
    #         plt.clf()
    #
    #     # with open(out_dir + '/' + ucca_file.rsplit('/', maxsplit=1)[1], 'w') as f:
    #
    #     for p in uconv.split_passage(passage, doc['ends'],
    #                                  map(lambda x: ''.join(x['sent_id'].split('-')[-2:]), doc['sents'])):
    #         uconv.passage2file(p, out_dir + '/' + p.ID + '.xml')
    #
    #     # print(f'on avg {len(unit_times)/sum(unit_times)} u/s', file=sys.stderr)
    #
    #
    # for x, y in tag_refinements.most_common(len(tag_refinements)):
    #     print(x, y, sep='\t')
    #
    #
    #
    #
    #
