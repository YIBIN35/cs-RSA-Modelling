import numpy as np
from updated_model import singleton_overspecification_rate, pair_overspecification_rate


def test_main_outputs():
    assert np.allclose(
        singleton_overspecification_rate(
            beta_fixed=1,
            state_semvalue_marked=0.9,
            state_semvalue_unmarked=0.9,
            costWeight=0,
        ),
        [0.31809810398844574, 0.31809810398844574],
    )
    assert np.allclose(
        singleton_overspecification_rate(
            beta_fixed=0,
            state_semvalue_marked=0.9,
            state_semvalue_unmarked=0.9,
            costWeight=0,
        ),
        [0.9140044558182415, 0.4792888453302001],
    )
    assert np.allclose(
        singleton_overspecification_rate(
            beta_fixed=0.69,
            state_semvalue_marked=0.9,
            state_semvalue_unmarked=0.9,
            costWeight=0,
        ),
        [0.5337370137540608, 0.36616687414861154],
    )
    assert np.allclose(
        singleton_overspecification_rate(
            beta_fixed=0.69,
            state_semvalue_marked=0.9,
            state_semvalue_unmarked=0.9,
            costWeight=1.5,
        ),
        [0.20345373360204588, 0.11418415477372575],
    )
    assert np.allclose(
        singleton_overspecification_rate(
            beta_fixed=0.69,
            state_semvalue_marked=0.9,
            state_semvalue_unmarked=0.9,
            costWeight=3,
        ),
        [0.053918932642215554, 0.02795797520941626],
    )

    assert np.allclose(
        pair_overspecification_rate(
            beta_fixed=1,
            state_semvalue_marked=0.9,
            state_semvalue_unmarked=0.9,
            costWeight=0,
        ),
        [0.3379059891857068, 0.3379059891857068],
    )
    assert np.allclose(
        pair_overspecification_rate(
            beta_fixed=0,
            state_semvalue_marked=0.9,
            state_semvalue_unmarked=0.9,
            costWeight=0,
        ),
        [0.7688537513814253, 0.4901187491353383],
    )
    assert np.allclose(
        pair_overspecification_rate(
            beta_fixed=0.69,
            state_semvalue_marked=0.9,
            state_semvalue_unmarked=0.9,
            costWeight=0,
        ),
        [0.49182432890881916, 0.3812474501902033],
    )
    assert np.allclose(
        pair_overspecification_rate(
            beta_fixed=0.69,
            state_semvalue_marked=0.9,
            state_semvalue_unmarked=0.9,
            costWeight=1.5,
        ),
        [0.17759816913487325, 0.12086579207614231],
    )
