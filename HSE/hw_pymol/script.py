phi = -60
psi = -45
cmd.fragment('ala')
for x in range(1, 25):
    cmd.edit(f'resi {x + 1} and name c')
    editor.attach_amino_acid('pk1', 'ala')
    cmd.edit(f'resi {x + 2} and name n',  f'resi {x + 2} and name ca')
    cmd.torsion(phi)
    cmd.edit( f'resi {x + 2} and name ca', f'resi {x + 2} and name c')
    cmd.torsion(psi)
