"""Small numerical contracts for the estimators used by the report."""
from pathlib import Path
import sys,unittest
import numpy as np
from scipy.optimize import approx_fprime
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ai_learning.data import bh_adjust,rcs
from ai_learning.firth import components,fit_firth
from ai_learning.prediction import losses

class NumericalContracts(unittest.TestCase):
    def test_bh_missing_test_remains_in_scope(self):
        actual=bh_adjust([.01,.03,np.nan,.2])
        np.testing.assert_allclose(actual[[0,1,3]],[.04,.06,.2*4/3])
        self.assertTrue(np.isnan(actual[2]))

    def test_adjusted_score_matches_numerical_gradient(self):
        rng=np.random.default_rng(12);X=np.column_stack([np.ones(90),rng.normal(size=(90,3))])
        y=rng.binomial(1,.5,size=90);beta=np.array([.1,-.2,.15,.3]);w=np.ones(90)
        score=components(beta,X,y,w)[1]
        numeric=approx_fprime(beta,lambda b:components(b,X,y,w)[0],1e-7)
        np.testing.assert_allclose(score,numeric,atol=2e-6)

    def test_weighted_and_physical_resampling_agree(self):
        rng=np.random.default_rng(31);X=np.column_stack([np.ones(90),rng.normal(size=(90,2))]);y=rng.binomial(1,.4,90)
        ix=rng.integers(0,90,90);w=np.bincount(ix,minlength=90)
        a=fit_firth(X,y,weights=w);b=fit_firth(X[ix],y[ix])
        np.testing.assert_allclose(a['beta'],b['beta'],atol=2e-6)

    def test_spline_derivative(self):
        z=np.array([-.7,.2,.8,1.4]);knots=np.array([-1.,0.,1.]);h=1e-6
        numeric=(rcs(z+h,knots)-rcs(z-h,knots))/(2*h)
        np.testing.assert_allclose(rcs(z,knots,True),numeric,atol=2e-6)

    def test_multiclass_loss_definitions(self):
        y=np.array([0,1,2]);p=np.full((3,3),.1);p[np.arange(3),y]=.8
        scores=losses(y,p)
        self.assertAlmostEqual(scores['log_loss'].mean(),-np.log(.8))
        self.assertAlmostEqual(scores['Brier'].mean(),.06)
        self.assertAlmostEqual(scores['RPS'].mean(),.02)

if __name__=='__main__':unittest.main()
