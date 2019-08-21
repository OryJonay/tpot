"""Tests that ensure the dask-based fit matches.

https://github.com/DEAP/deap/issues/75
"""
import unittest
from contextlib import contextmanager
import nose
from sklearn.datasets import make_classification

from tpot import TPOTClassifier

try:
    import dask  # noqa
    import dask_ml  # noqa
    import distributed  # noqa
except ImportError:
    raise nose.SkipTest()

@contextmanager
def local_cluster_context():
    try:
        client = distributed.Client(processes=False, threads_per_worker=1)
        distributed.client._set_global_client(client)
        yield
    except Exception:
        raise
    finally:
        distributed.client._del_global_client(client)

class TestDaskMatches(unittest.TestCase):

    def test_dask_matches(self):
        with local_cluster_context():
            with dask.config.set(scheduler='dask.distributed'):
                for n_jobs in [-1]:
                    X, y = make_classification(random_state=42)
                    a = TPOTClassifier(
                        generations=0,
                        population_size=5,
                        cv=3,
                        random_state=42,
                        n_jobs=n_jobs,
                        use_dask=False
                    )
                    b = TPOTClassifier(
                        generations=0,
                        population_size=5,
                        cv=3,
                        random_state=42,
                        n_jobs=n_jobs,
                        use_dask=True
                    )
                    a.fit(X, y)
                    b.fit(X, y)

                    self.assertEqual(a.score(X, y), b.score(X, y))
                    self.assertEqual(a.pareto_front_fitted_pipelines_.keys(),
                                     b.pareto_front_fitted_pipelines_.keys())
                    self.assertEqual(a.evaluated_individuals_,
                                     b.evaluated_individuals_)
