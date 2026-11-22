import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svd_recommender import SVDRecommender, rating_matrix, metrics


class SVDTests(unittest.TestCase):
    def setUp(self):
        self.train = np.array([[5, 4, np.nan, 1], [4, np.nan, 2, 1], [1, 2, 5, np.nan]], float)
        self.model = SVDRecommender().fit(self.train)

    def test_training_statistics_and_imputation(self):
        np.testing.assert_allclose(self.model.user_mean, [10/3, 7/3, 8/3])
        self.assertEqual(self.model.filled[0, 2], 3.5)
        np.testing.assert_allclose(self.model.filled[self.model.observed], self.train[self.model.observed])
        np.testing.assert_allclose(np.where(self.model.observed, self.model.centered, 0).sum(1), 0, atol=1e-14)

    def test_full_rank_returns_imputation_and_known_values(self):
        np.testing.assert_allclose(self.model.reconstruct(3, clip=False), self.model.filled, atol=1e-12)

    def test_factors_prediction_and_eckart_young_identity(self):
        for k in range(4):
            p, q = self.model.factors(k)
            residual = p @ q.T
            np.testing.assert_allclose(self.model.reconstruct(k, clip=False), self.model.user_mean[:, None] + residual, atol=1e-12)
            loss = np.linalg.norm(self.model.centered-residual, 'fro')**2
            self.assertAlmostEqual(loss, float(np.sum(self.model.s[k:]**2)), places=10)
        users, items = np.indices(self.train.shape)
        np.testing.assert_allclose(self.model.predict(users, items, 2), self.model.reconstruct(2))
        self.assertAlmostEqual(float(self.model.predict(0, 2, 2)), self.model.reconstruct(2)[0, 2])

    def test_cold_start_clipping_and_recommendations(self):
        train = np.array([[5, np.nan, np.nan], [1, 2, np.nan], [np.nan, np.nan, np.nan]])
        model = SVDRecommender().fit(train)
        self.assertEqual(float(model.predict(2, 0, 1)), 3)
        self.assertEqual(float(model.predict(0, 2, 1)), 5)
        self.assertTrue(np.isfinite(model.reconstruct(1)).all())
        for k in range(4):
            pred = self.model.reconstruct(k)
            self.assertTrue(np.all((pred >= 1) & (pred <= 5)))
        self.assertEqual([i for i, _ in self.model.recommend(0, 2)], [2])

    def test_validation_rejections_and_metrics(self):
        with self.assertRaises(ValueError):
            rating_matrix([[0, 0, 5], [0, 0, 4]], (2, 2))
        with self.assertRaises(ValueError):
            SVDRecommender().fit(np.array([[0, 5]]))
        with self.assertRaises(ValueError):
            self.model.reconstruct(4)
        self.assertEqual(metrics([1, 5], [2, 4]), {"rmse": 1, "mae": 1})


if __name__ == '__main__':
    unittest.main()
