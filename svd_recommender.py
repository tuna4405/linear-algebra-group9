"""Dense, mean-imputed truncated SVD. No recommendation libraries are used."""
from dataclasses import dataclass
import numpy as np


def rating_matrix(triples, shape):
    """Convert zero-based (user, item, rating) records to a NaN-masked matrix."""
    triples = np.asarray(triples, dtype=float)
    if triples.ndim != 2 or triples.shape[1] != 3 or not len(triples):
        raise ValueError("Expected nonempty (user, item, rating) records")
    indices = triples[:, :2]
    if not np.all(np.isfinite(triples)) or np.any(indices != np.floor(indices)):
        raise ValueError("Indices must be integers and ratings must be finite")
    users, items = indices.astype(int).T
    if np.any(users < 0) or np.any(users >= shape[0]) or np.any(items < 0) or np.any(items >= shape[1]):
        raise ValueError("User or item index outside matrix shape")
    if len(np.unique(users * shape[1] + items)) != len(triples):
        raise ValueError("Duplicate user-item ratings must be resolved explicitly")
    matrix = np.full(shape, np.nan)
    matrix[users, items] = triples[:, 2]
    return matrix


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or not actual.size:
        raise ValueError("Metric inputs must be nonempty and have equal shape")
    error = predicted - actual
    if not np.all(np.isfinite(error)):
        raise ValueError("Metric inputs must be finite")
    return {"rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error)))}


@dataclass
class SVDRecommender:
    imputation: str = "item"
    rating_min: float = 1.0
    rating_max: float = 5.0

    def fit(self, train):
        """Fit ONLY to training entries; NaN denotes an unobserved rating.

        Missing raw entries receive the training item mean (default) or user
        mean. Every row is then shifted by its OBSERVED training user mean.
        An empty training user/item receives the global training mean.
        """
        train = np.array(train, dtype=float, copy=True)
        if train.ndim != 2 or not all(train.shape):
            raise ValueError("Expected a nonempty 2D matrix")
        if self.imputation not in {"item", "user"}:
            raise ValueError("Imputation must be 'item' or 'user'")
        if self.rating_min >= self.rating_max:
            raise ValueError("Invalid rating range")
        observed = ~np.isnan(train)
        values = train[observed]
        if not values.size or not np.all(np.isfinite(values)):
            raise ValueError("Training data must contain finite observed ratings")
        if np.any(values < self.rating_min) or np.any(values > self.rating_max):
            raise ValueError("Observed rating outside valid range; use NaN for missing entries")
        self.train = train
        self.observed = observed
        self.global_mean = float(values.mean())
        self.user_count = observed.sum(axis=1)
        self.item_count = observed.sum(axis=0)
        self.user_mean = np.divide(np.nansum(train, axis=1), self.user_count,
                                   out=np.full(train.shape[0], self.global_mean), where=self.user_count > 0)
        self.item_mean = np.divide(np.nansum(train, axis=0), self.item_count,
                                   out=np.full(train.shape[1], self.global_mean), where=self.item_count > 0)
        prior = self.item_mean[None, :] if self.imputation == "item" else self.user_mean[:, None]
        self.filled = np.where(observed, train, prior)
        self.centered = self.filled - self.user_mean[:, None]
        self.U, self.s, self.Vt = np.linalg.svd(self.centered, full_matrices=False)
        return self

    def _check_k(self, k):
        if not isinstance(k, (int, np.integer)) or not 0 <= k <= len(self.s):
            raise ValueError(f"k must be an integer from 0 to {len(self.s)}")

    def factors(self, k):
        """P: users x k, Q: items x k; P @ Q.T reconstructs deviations."""
        self._check_k(k)
        root = np.sqrt(self.s[:k])
        return self.U[:, :k] * root, self.Vt[:k].T * root

    def predict(self, users, items, k, clip=True):
        self._check_k(k)
        users, items = np.broadcast_arrays(np.asarray(users, dtype=int), np.asarray(items, dtype=int))
        if np.any(users < 0) or np.any(users >= self.train.shape[0]) or np.any(items < 0) or np.any(items >= self.train.shape[1]):
            raise ValueError("Unknown ID; use a known matrix row/column, including an empty one")
        # Flattening supports scalar and multidimensional broadcast inputs.
        u, i = users.ravel(), items.ravel()
        residual = np.sum(self.U[u, :k] * self.s[:k] * self.Vt[:k, i].T, axis=1).reshape(users.shape)
        pred = self.user_mean[users] + residual
        # Cold users cannot have a learned personal taste; cold items have no
        # observed item signal. Use explicit baselines rather than SVD artifacts.
        pred = np.where(self.user_count[users] == 0, self.item_mean[items], pred)
        pred = np.where(self.item_count[items] == 0, self.user_mean[users], pred)
        return np.clip(pred, self.rating_min, self.rating_max) if clip else pred

    def reconstruct(self, k, clip=True):
        self._check_k(k)
        pred = self.user_mean[:, None] + (self.U[:, :k] * self.s[:k]) @ self.Vt[:k]
        pred[self.user_count == 0, :] = self.item_mean
        pred[:, self.item_count == 0] = self.user_mean[:, None]
        return np.clip(pred, self.rating_min, self.rating_max) if clip else pred

    def recommend(self, user, k, top_n=5):
        """Rank unrated items; this is a demo, not a ranking-quality evaluation."""
        if not 0 <= user < self.train.shape[0] or top_n < 1:
            raise ValueError("Invalid user or top_n")
        candidates = np.flatnonzero(~self.observed[user])
        scores = self.predict(np.full(len(candidates), user), candidates, k)
        order = np.argsort(-scores, kind="stable")[:top_n]
        return [(int(candidates[j]), float(scores[j])) for j in order]
