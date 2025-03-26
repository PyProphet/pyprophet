from functools import reduce
from collections import OrderedDict

from .modmass import ModifiedPeptideMassCalculator, ModInfo, ModSite
from .pepmass import FragmentTypeInfo

class GlycanNode:
    def __init__(self, monosaccharide, children=None):
        self.monosaccharide = monosaccharide
        self.children = children

    def __str__(self):
        s = '(' + self.monosaccharide
        if isinstance(self.children, list):
            s = s + ''.join(str(c) for c in self.children)
        elif self.children is not None:
            s = s + str(self.children)
        return s + ')'

    def composition(self):
        def add(d1, d2):
            return {
                k: d1.get(k, 0) + d2.get(k, 0)
                for k in set(d1) | set(d2)
            }

        d = { self.monosaccharide: 1 }
        if isinstance(self.children, list):
            d = reduce(add, (c.composition() for c in self.children), d)
        elif self.children is not None:
            d = add(d, self.children.composition())
        return d

    def composition_str(self):
        composition = self.composition()

        return ''.join((
            str(k) + '(' + str(v) +')'
            for k, v in sorted(composition.items(), key=lambda t: t[0])
        ))


    @staticmethod
    def from_str(s):
        if s.startswith('(') and s.endswith(')'):
            s = s[1:-1]

        nodes = list()
        start = None
        for i, c in enumerate(s):
            if c != '(' and c != ')':
                if start is None:
                    start = i
            else:
                if start is not None:
                    node = GlycanNode(s[start:i])
                    if len(nodes) > 0:
                        if isinstance(nodes[-1].children, list):
                            nodes[-1].children.append(node)
                        elif nodes[-1].children is None:
                            nodes[-1].children = [node]
                        else:
                            nodes[-1].children = [nodes[-1].children, node]
                    nodes.append(node)
                    start = None

            if c == ')':
                if len(nodes) <= 1:
                    raise ValueError('invalid format: ' + s)
                nodes.pop()

        if len(nodes) == 0 and start is not None:
            return GlycanNode(s[start:])
        elif len(nodes) == 1:
            return nodes[0]
        else:
            raise ValueError('invalid format: ' + s)



class GlycoPeptideMassCalculator(ModifiedPeptideMassCalculator):
    def __init__(self, monosaccharide=None, fragments=None,
                 **kwargs):
        if fragments is None:
            fragments = {
                'b': FragmentTypeInfo.b(),
                'y': FragmentTypeInfo.y()
            }

        super(GlycoPeptideMassCalculator, self) \
            .__init__(fragments=fragments, **kwargs)

        self.aa_residues.setdefault(
            'J',
            self.aa_residue_mass('N')
        )

        if monosaccharide is None:
            monosaccharide = {
                'H': 162.0528234, # Hex    C(6)H(10)O(5)
                'N': 203.0793725, # HexNAc C(8)H(13)N(1)O(5)
                'A': 291.0954165, # NeuAc  C(11)H(17)N(1)O(8)
                'G': 307.0903311, # NeuGc  C(11)H(17)N(1)O(9)
                'F': 146.0579088, # dHex   C(6)H(10)O(4)
                'X': 132.0422588, # Xyl    C(5)H(8)O(4)
                'R': 176.0320881  # HexA   C(6)H(8)O(6)
            }

        self.monosaccharide = monosaccharide

        if not any(filter(
            lambda x: x.name == 'GlycoMod-N(1)' and x.site == 'J',
            self.variable_modifications
        )):
            self.variable_modifications.append(
                ModInfo(
                    name='GlycoMod-N(1)',
                    site=['J', 'S', 'T'],
                    delta_mass=self.monosaccharide_mass('N')
                )
            )
        if not any(filter(
            lambda x: x.name == 'GlycoMod$' and x.site == 'J',
            self.variable_modifications
        )):
            self.variable_modifications.append(
                ModInfo(
                    name='GlycoMod$',
                    site=['J', 'S', 'T'],
                    delta_mass=self.element_mass('C') * 4 +
                        self.element_mass('H') * 5 +
                        self.element_mass('N') + self.element_mass('O')
                )
            )

        oxonium_ions = kwargs.get('oxonium_ions', None)
        if oxonium_ions is None:
            oxonium_ions = [
                {
                    'monosaccharide': 'H',
                    'name': 'C6H4O2',
                    'mass': self.element_mass('C') * 6 + \
                        self.element_mass('H') * 4 + \
                        self.element_mass('O') * 2
                },
                {
                    'monosaccharide': 'H',
                    'name': 'C5H6O3',
                    'mass': self.element_mass('C') * 5 + \
                        self.element_mass('H') * 6 + \
                        self.element_mass('O') * 3
                },
                {
                    'monosaccharide': 'N',
                    'name': 'C6H7O2N1',
                    'mass': self.element_mass('C') * 6 + \
                        self.element_mass('H') * 7 + \
                        self.element_mass('O') * 2 + \
                        self.element_mass('N')
                },
                {
                    'monosaccharide': 'H',
                    'name': 'C6H6O3',
                    'mass': self.element_mass('C') * 6 + \
                        self.element_mass('H') * 6 + \
                        self.element_mass('O') * 3
                },
                {
                    'monosaccharide': 'N',
                    'name': 'C7H7O2N1',
                    'mass': self.element_mass('C') * 7 + \
                        self.element_mass('H') * 7 + \
                        self.element_mass('O') * 2 + \
                        self.element_mass('N')
                },
                {
                    'monosaccharide': 'N',
                    'name': 'C6H9O3N1',
                    'mass': self.element_mass('C') * 6 + \
                        self.element_mass('H') * 9 + \
                        self.element_mass('O') * 3 + \
                        self.element_mass('N')
                },
                {
                    'monosaccharide': 'H',
                    'name': 'C6H10O5',
                    'mass': self.element_mass('C') * 6 + \
                        self.element_mass('H') * 10 + \
                        self.element_mass('O') * 5
                },
                {
                    'monosaccharide': 'N',
                    'name': 'C8H9O3N1',
                    'mass': self.element_mass('C') * 8 + \
                        self.element_mass('H') * 9 + \
                        self.element_mass('O') * 3 + \
                        self.element_mass('N')
                },
                {
                    'monosaccharide': 'N',
                    'name': 'C8H11O4N1',
                    'mass': self.element_mass('C') * 8 + \
                        self.element_mass('H') * 11 + \
                        self.element_mass('O') * 4 + \
                        self.element_mass('N')
                },
                {
                    'monosaccharide': 'N',
                    'name': 'C8H13O5N1',
                    'mass': self.element_mass('C') * 8 + \
                        self.element_mass('H') * 13 + \
                        self.element_mass('O') * 5 + \
                        self.element_mass('N')
                },
                {
                    'monosaccharide': 'A',
                    'name': 'C11H15O7N1',
                    'mass': self.element_mass('C') * 11 + \
                        self.element_mass('H') * 15 + \
                        self.element_mass('O') * 7 + \
                        self.element_mass('N')
                },
                {
                    'monosaccharide': 'G',
                    'name': 'C11H5O8N1',
                    'mass': self.element_mass('C') * 11 + \
                        self.element_mass('H') * 15 + \
                        self.element_mass('O') * 8 + \
                        self.element_mass('N')
                },
                {
                    'monosaccharide': 'A',
                    'name': 'C11H17O8N1',
                    'mass': self.element_mass('C') * 11 + \
                        self.element_mass('H') * 17 + \
                        self.element_mass('O') * 8 + \
                        self.element_mass('N')
                },
                {
                    'monosaccharide': 'G',
                    'name': 'C11H17O9N1',
                    'mass': self.element_mass('C') * 11 + \
                        self.element_mass('H') * 17 + \
                        self.element_mass('O') * 9 + \
                        self.element_mass('N')
                },
                {
                    'monosaccharide': ['H', 'N'],
                    'name': 'C14H23O10N1',
                    'mass': self.element_mass('C') * 14 + \
                        self.element_mass('H') * 23 + \
                        self.element_mass('O') * 10 + \
                        self.element_mass('N')
                },
            ]

        self.oxonium_ions = oxonium_ions


    def monosaccharide_mass(self, monosaccharide):
        result = self.monosaccharide.get(monosaccharide, None)
        if result is None:
            raise ValueError(
                'unknown monosaccharide: ' + str(monosaccharide)
            )
        return result

    def glycan_mw(self, glycan):
        if isinstance(glycan, str):
            glycan = GlycanNode.from_str(glycan)

        return sum(
            self.monosaccharide_mass(k) * v
            for k, v in glycan.composition().items()
        )


    def mw(self, sequence, glycan=None, **kwargs):
        result = super(GlycoPeptideMassCalculator, self) \
            .mw(sequence=sequence, **kwargs)

        if glycan is not None:
            result += self.glycan_mw(glycan)

        return result


    def glycan_fragment(self, glycan):
        def _glycan_fragment(glycan, allow_duplicate=False):
            def concat(a, b):
                return a + b

            def add(d1, d2):
                return {
                    k: d1.get(k, 0) + d2.get(k, 0)
                    for k in set(d1) | set(d2)
                }

            def cross_add(l1, l2):
                return [
                    add(x1, x2)
                    for x1 in l1
                    for x2 in l2
                ]

            d = { glycan.monosaccharide: 1 }
            if isinstance(glycan.children, list):
                r = list(map(lambda x: add(d, x), reduce(cross_add, (
                    concat([{}], _glycan_fragment(c))
                    for c in glycan.children
                ))))
            elif glycan.children is not None:
                r = concat([d], list(map(
                    lambda x: add(d, x),
                    _glycan_fragment(glycan.children)
                )))
            else:
                r = [d]

            if not allow_duplicate:
                r = list(OrderedDict((
                    frozenset(d.items()), d)
                    for d in r
                ).values())
            return r

        if isinstance(glycan, str):
            glycan = GlycanNode.from_str(glycan)
        if glycan is None:
            raise ValueError(
                'invalid glycan: ' + str(glycan)
            )

        fragment = _glycan_fragment(glycan)
        fragment.remove(glycan.composition())
        return fragment


    def glycan_fragment_mw(self, sequence, glycan, **kwargs):
        def _glycan_fragment_mw(fragment):
            return sum(
                self.monosaccharide_mass(k) * v
                for k, v in fragment.items()
            )

        def _glycan_fragment_name(fragment):
            return ''.join(
                k + '(' + str(v) + ')'
                for k, v in fragment.items()
            )

        seq_mw = super(GlycoPeptideMassCalculator, self) \
            .mw(sequence=sequence, **kwargs)

        fragment = self.glycan_fragment(glycan)

        fragment_mw = [
            seq_mw + _glycan_fragment_mw(x)
            for x in fragment
        ]
        fragment_name = [
            'Y-' + _glycan_fragment_name(x)
            for x in fragment
        ]

        fragment_mw.insert(0, seq_mw)
        fragment_name.insert(0, 'Y0')

        mod = next((
            mod for mod in self.variable_modifications
            if mod.name == 'GlycoMod$'
        ), None)
        if mod is not None:
            fragment_mw.append(seq_mw + mod.delta_mass)
            fragment_name.append('Y$')

        return {
            'fragment_mw': fragment_mw,
            'fragment_name': fragment_name,
            'fragment_type': 'Y'
        }


    def peptide_fragment_mw(self, sequence, modification=None,
                            fragment_type='b', glycan_site=None,
                            **kwargs):
        def _parse_fragment_type(fragment_type, allow_list=False):
            if isinstance(fragment_type, str):
                if fragment_type in self.fragments:
                    return {'naked': [fragment_type]}
                for frag in self.fragments:
                    if fragment_type.startswith(frag):
                        return {fragment_type[len(frag):]: [frag]}
                raise ValueError(
                    'unknown fragment type: ' + fragment_type
                )
            elif allow_list and \
                (isinstance(fragment_type, list) or \
                isinstance(fragment_type, tuple)):
                def merge(d1, d2):
                    return {
                        k: d1.get(k, []) + d2.get(k, [])
                        for k in set(d1) | set(d2)
                    }
                return reduce(merge, map(_parse_fragment_type, fragment_type))
            else:
                raise TypeError('invalid fragment type: ' + \
                    str(type(fragment_type)))

        fragment_type_dict = _parse_fragment_type(fragment_type, allow_list=True)

        fragment_mw = []

        nake_type = fragment_type_dict.pop('naked', None)
        if nake_type is not None:
            fragment_mw = super(GlycoPeptideMassCalculator, self) \
                .fragment_neutral_mw(
                    sequence=sequence,
                    modification=modification,
                    fragment_type=nake_type,
                    **kwargs
                )

        if len(fragment_type_dict) > 0:
            def match_record(x, lst):
                for i, y in enumerate(lst):
                    if all(y.get(k, v) == v for k, v in x.items()):
                        return i, y
                return None, None

            if glycan_site is not None:
                glyco_position = glycan_site
            else:
                glyco_position = sequence.find('J') + 1
            if (glyco_position <= 0):
                raise ValueError('Glyco site not found in ' + sequence)

            for k, v in fragment_type_dict.items():
                glyco_mod_site = ModSite(
                    name='GlycoMod' + k,
                    position=glyco_position,
                    site=sequence[glyco_position - 1]
                )
                if isinstance(modification, list):
                    new_modification = modification + [glyco_mod_site]
                elif modification is not None:
                    new_modification = [modification, glyco_mod_site]
                else:
                    new_modification = glyco_mod_site

                new_fragment_mw = super(GlycoPeptideMassCalculator, self) \
                    .fragment_neutral_mw(
                        sequence=sequence,
                        modification=new_modification,
                        fragment_type=v,
                        **kwargs
                    )
                glycosite_count = self.fragment_mod_count(
                    sequence=sequence,
                    modification=glyco_mod_site,
                    fragment_type=v,
                )

                for i in range(len(new_fragment_mw)):
                    glycosite_count_i = match_record(
                        new_fragment_mw[i],
                        glycosite_count
                    )[1]['mod_count']

                    for mod, count in glycosite_count_i.items():
                        if mod.name == 'GlycoMod' + k and mod.site == 'J':
                            for j in range(len(count)):
                                if count[j] == 0:
                                    new_fragment_mw[i]['fragment_mw'][j] = None

                    new_fragment_mw[i]['fragment_type'] += k


                fragment_mw.extend(new_fragment_mw)

        if isinstance(fragment_type, str):
            fragment_mw = fragment_mw[0]
        return fragment_mw


    def fragment_neutral_mw(self, sequence, modification=None,
                            glycan=None,
                            fragment_type='b',
                            **kwargs):
        def _parse_fragment_type(fragment_type, allow_list=False):
            if fragment_type == 'Y':
                return None, fragment_type
            elif isinstance(fragment_type, str):
                return fragment_type, None
            elif allow_list and \
                (isinstance(fragment_type, list) or \
                isinstance(fragment_type, tuple)):
                l = list(map(_parse_fragment_type, fragment_type))
                peptide_type = [
                    t[0] for t in l if t[0] is not None
                ]
                if len(peptide_type) == 0:
                    peptide_type = None
                glyco_type = [
                    t[1] for t in l if t[1] is not None
                ]
                if len(glyco_type) == 0:
                    glyco_type = None
                return peptide_type, glyco_type

        peptide_type, glyco_type = _parse_fragment_type(
            fragment_type, allow_list=True
        )

        fragment_mw = []
        if peptide_type is not None:
            peptide_fragment_mw = self.peptide_fragment_mw(
                sequence=sequence,
                modification=modification,
                fragment_type=peptide_type,
                **kwargs
            )
            if isinstance(peptide_fragment_mw, list) and \
                len(peptide_fragment_mw) > 0 and \
                isinstance(peptide_fragment_mw[0], dict):
                fragment_mw.extend(peptide_fragment_mw)
            else:
                fragment_mw.append(peptide_fragment_mw)

        if glyco_type is not None:
            glyco_fragment_mw = self.glycan_fragment_mw(
                sequence=sequence,
                modification=modification,
                glycan=glycan,
                fragment_type=glyco_type,
                **kwargs
            )
            if isinstance(glyco_fragment_mw, list) and \
                len(glyco_fragment_mw) > 0 and \
                isinstance(glyco_fragment_mw[0], dict):
                fragment_mw.extend(glyco_fragment_mw)
            else:
                fragment_mw.append(glyco_fragment_mw)

        if isinstance(fragment_type, str):
            fragment_mw = fragment_mw[0]
        return fragment_mw


    def oxonium_ion_mz(self, monosaccharide=None, glycan=None):
        if isinstance(monosaccharide, str):
            monosaccharide = [monosaccharide]

        if isinstance(glycan, str):
            glycan = GlycanNode.from_str(glycan)

        if glycan is not None:
            monosaccharide = glycan.composition().keys()

        fragment_mz = []
        fragment_name = []
        for oxonium in self.oxonium_ions:
            keep = False
            if monosaccharide is None:
                keep = True
            elif isinstance(oxonium['monosaccharide'], str):
                if oxonium['monosaccharide'] in monosaccharide:
                    keep = True
            elif isinstance(oxonium['monosaccharide'], list):
                if all(map(lambda x: x in monosaccharide, \
                           oxonium['monosaccharide'])):
                    keep = True

            if keep:
                fragment_mz.append(oxonium['mass'] + \
                                   self.element_mass('proton'))
                fragment_name.append('+'.join(oxonium['monosaccharide']) + \
                                     ':' + oxonium['name'])

        return {
            'fragment_mz': fragment_mz,
            'fragment_name': fragment_name,
            'fragment_type': 'oxonium'
        }



if __name__ == '__main__':
    pep_calc = GlycoPeptideMassCalculator()

    #print(pep_calc.glycan_fragment('(N(N(H(H(N(H(A))))(H(N(H(A)))))))'))

    print(pep_calc.glycan_fragment_mw(
       sequence='LGJWSAMPSCK',
       glycan='(N(N(H(H(N(H(A))))(H(N(H(A)))))))'
    ))

    print(pep_calc.peptide_fragment_mw(
        sequence='LGJWSAMPSCK',
        fragment_type='y-N(1)'
    ))

    print(pep_calc.fragment_neutral_mw(
        sequence='LGJWSAMPSCK',
        glycan='(N(N(H(H(N(H(A))))(H(N(H(A)))))))',
        fragment_type=['b', 'y', 'Y', 'y$']
    ))

    print(pep_calc.fragment_neutral_mw(
        sequence='DAHKSEVAHR',
        glycan='(H)', glycan_site=5,
        fragment_type=['b', 'y', 'Y', 'y$']
    ))

