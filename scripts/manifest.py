"""Dataset manifest: every structure in the training/test set, reproducibly."""


def build_manifest():
    m = []

    def add(tag, **kw):
        m.append((f'{tag}_{len(m):03d}', kw))

    # pristine, rattled
    add('prist', n_vac=0, rattle=0.0, seed=1)
    for s in range(2, 6):
        add('prist', n_vac=0, rattle=0.02, seed=s)
    for s in range(6, 10):
        add('prist', n_vac=0, rattle=0.04, seed=s)
    for s in range(10, 12):
        add('prist', n_vac=0, rattle=0.06, seed=s)
    # one vacancy, top layer
    add('vac1', n_vac=1, rattle=0.0, seed=20)
    for s in range(21, 27):
        add('vac1', n_vac=1, rattle=0.02, seed=s)
    for s in range(27, 32):
        add('vac1', n_vac=1, rattle=0.04, seed=s)
    # one vacancy, bottom layer
    for s in range(32, 36):
        add('vac1b', n_vac=1, rattle=0.02, seed=s, vac_layer='bottom')
    # two vacancies (varied separations via seeds), top and mixed layers
    add('vac2', n_vac=2, rattle=0.0, seed=40)
    for s in range(41, 48):
        add('vac2', n_vac=2, rattle=0.02, seed=s)
    for s in range(48, 52):
        add('vac2', n_vac=2, rattle=0.04, seed=s)
    for s in range(52, 54):
        add('vac2m', n_vac=2, rattle=0.02, seed=s, vac_layer='any')
    # three vacancies
    add('vac3', n_vac=3, rattle=0.0, seed=60)
    for s in range(61, 66):
        add('vac3', n_vac=3, rattle=0.02, seed=s, vac_layer='any')
    # strain
    for s, st in [(70, -0.01), (71, -0.01), (72, 0.01), (73, 0.01)]:
        add('strp', n_vac=0, rattle=0.02, seed=s, strain=st)
    for s, st in [(74, -0.01), (75, -0.01), (76, 0.01), (77, 0.01)]:
        add('strv', n_vac=1, rattle=0.02, seed=s, strain=st)
    return m


def build_test_manifest():
    m = []

    def add(tag, **kw):
        m.append((f'test_{tag}_{len(m):03d}', kw))

    add('prist', n_vac=0, rattle=0.03, seed=101)
    add('vac1', n_vac=1, rattle=0.03, seed=102)
    add('vac2', n_vac=2, rattle=0.03, seed=103)
    add('vac2', n_vac=2, rattle=0.02, seed=104, vac_layer='any')
    add('vac3', n_vac=3, rattle=0.02, seed=105, vac_layer='any')
    add('strv', n_vac=1, rattle=0.02, seed=106, strain=0.008)
    return m


if __name__ == '__main__':
    tr, te = build_manifest(), build_test_manifest()
    print(len(tr), 'train structures,', len(te), 'test structures')
    for name, kw in tr[:5] + te[:3]:
        print(name, kw)
