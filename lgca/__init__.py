def get_lgca(geometry='hex',  **kwargs):
    if geometry in ['1d', 'lin', 'linear']:
        from .lgca_1d import LGCA_1D
        return LGCA_1D(**kwargs)

    elif geometry in ['square', 'sq', 'rect', 'rectangular']:
        from .lgca_square import LGCA_Square
        return LGCA_Square(**kwargs)

    else:
        from .lgca_hex import LGCA_Hex
        return LGCA_Hex(**kwargs)
