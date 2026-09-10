import numpy as np
import pytest
from scripts.analysis.visbal_duty_uvlf import active_weights, uv_coefficient


def test_occupation_conserves_number_and_mean_light():
    w=np.array([1.,2.,3.]); eligible=np.array([True,False,True])
    for duty in (.01,.1,1.):
        a,b=active_weights(w,eligible,duty)
        np.testing.assert_allclose(a+b,w)
        assert np.sum(a/duty)==4.


def test_constant_ssp_units():
    assert uv_coefficient(np.array([.01,1.,100.]),np.ones(3)*2,10.) == pytest.approx(2e7)


def test_invalid_ranges():
    with pytest.raises(ValueError):
        active_weights(np.ones(2),np.ones(2,dtype=bool),0)
    with pytest.raises(ValueError):
        uv_coefficient(np.array([1.,10.]),np.ones(2),100)


def test_batch_mean_does_not_multiply_density():
    from scripts.analysis.combine_visbal_batches import combine_window
    y,se=combine_window(np.array([3.,4.]),np.array([[1.,2.],[1.,2.],[1.,2.],[1.,2.]]))
    np.testing.assert_array_equal(y,[4.,6.])
    np.testing.assert_array_equal(se,[0.,0.])
