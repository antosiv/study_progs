import numpy as np


class MyKNeighborsClassifier:

    def __init__(self, n_neighbors=5):
        self.k = n_neighbors
        self.predict_data = None
        self.predict_class = None
        self.classes_numb = None

    def fit(self, x, y):
        self.predict_data = np.array(x, dtype=np.float32)
        self.predict_class = np.int8(y)
        self.classes_numb = np.unique(self.predict_class).shape[0]

    def predict(self, x):

        # classes of the nearest rows from
        # predict_data to X rows (shape: (X.shape[0], k))
        neigh_classes = self._neighbors_classes(x)
        res = np.empty((x.shape[0],), dtype=np.uint8)

        for i in range(x.shape[0]):
            res[i] = np.argmax(np.bincount(neigh_classes[i]))
        return res

    def predict_proba(self, x):

        # classes of the nearest rows from
        # predict_data to X rows (shape: (X.shape[0], k))
        neigh_classes = self._neighbors_classes(x)
        res = np.zeros((x.shape[0], self.classes_numb), dtype=np.float32)
        for i in range(x.shape[0]):
            counts = np.bincount(neigh_classes[i])
            res[i, :counts.size] = counts

        res /= self.k
        return res

    def score(self, x, y):
        tmp = self.predict(x)
        mistakes = np.sum(np.bincount(np.int8(np.abs(tmp - y)))[1:])
        return 1.0 - mistakes / y.shape[0]

    def _neighbors_classes(self, data):
        neighbors_classes = np.empty((data.shape[0],
                                      self.k), dtype=np.uint8)
        for i in range(data.shape[0]):
            neighbors_classes[i] = self.predict_class[
                np.argpartition(
                    ((np.array(data[i] - self.predict_data,
                               dtype=np.float32)) ** 2).sum(axis=1),
                    np.arange(self.k))[:self.k]
            ]
        return neighbors_classes


class MyDecisionTreeClassifier:
    NON_LEAF_TYPE = 0
    LEAF_TYPE = 1

    def __init__(self, min_samples_split=2, max_depth=None,
                 sufficient_share=1.0, criterion='gini', max_features=None
                 ):
        self.tree = dict()
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.sufficient_share = sufficient_share
        self.num_class = -1
        self.feature_importances_ = None
        if criterion == 'gini':
            self.G_function = self.__gini
        elif criterion == 'entropy':
            self.G_function = self.__entropy
        elif criterion == 'misclass':
            self.G_function = self.__misclass
        else:
            print('invalid criterion name')
            return

        if max_features == 'sqrt':
            self.get_feature_ids = self.__get_feature_ids_sqrt
        elif max_features == 'log2':
            self.get_feature_ids = self.__get_feature_ids_log2
        elif max_features is None:
            self.get_feature_ids = self.__get_feature_ids_N
        else:
            print('invalid max_features name')
            return

    def __gini(self, l_c, l_s, r_c, r_s):
        l_s = np.float32(l_s)
        r_s = np.float32(r_s)
        return np.array(
            (l_s / (l_s + r_s)) *
            (1.0 - (np.array(l_c / l_s, dtype=np.float32) ** 2)
             .sum(axis=-1)).reshape(-1, 1) +
            (r_s / (l_s + r_s)) *
            (1.0 - (np.array(r_c / r_s, dtype=np.float32) ** 2)
             .sum(axis=-1)).reshape(-1, 1), dtype=np.float32
        )

    def __entropy(self, l_c, l_s, r_c, r_s):
        l_s = np.float32(l_s)
        r_s = np.float32(r_s)

        # чтобы не делать лишних вычислений
        tmp_l = np.array(l_c / l_s, dtype=np.float32)
        tmp_r = np.array(r_c / r_s, dtype=np.float32)
        return -1.0 * (
                (l_s / (l_s + r_s)) * np.nan_to_num(tmp_l * np.log(tmp_l))
                .sum(axis=1).reshape(-1, 1) +
                (r_s / (l_s + r_s)) * np.nan_to_num(tmp_r * np.log(tmp_r))
                .sum(axis=1).reshape(-1, 1)
        )

    def __misclass(self, l_c, l_s, r_c, r_s):
        l_s = np.float32(l_s)
        r_s = np.float32(r_s)
        return (l_s / (l_s + r_s) *
                (1 - np.array(l_c / l_s, dtype=np.float32)
                 .max(axis=1)).reshape(-1, 1) +
                r_s / (l_s + r_s) *
                (1 - np.array(r_c / r_s, dtype=np.float32)
                 .max(axis=1)).reshape(-1, 1)
                )

    def __get_feature_ids_sqrt(self, n_feature):
        feature_ids = np.arange(n_feature)
        np.random.shuffle(feature_ids)
        return feature_ids[:np.sqrt(n_feature)]

    def __get_feature_ids_log2(self, n_feature):
        feature_ids = np.arange(n_feature)
        np.random.shuffle(feature_ids)
        return feature_ids[:np.log2(n_feature)]

    def __get_feature_ids_N(self, n_feature):
        return np.arange(n_feature)

    def __sort_samples(self, x, y):
        sorted_idx = x.argsort()
        return x[sorted_idx], y[sorted_idx]

    def __div_samples(self, x, y, feature_id, threshold):
        left_mask = x[:, feature_id] <= threshold
        right_mask = ~left_mask
        return x[left_mask], x[right_mask], y[left_mask], y[right_mask]

    def __find_threshold(self, x, y):

        sorted_x, sorted_y = self.__sort_samples(x, y)

        class_number = np.unique(y).shape[0]

        max_class = np.max(y) + 1

        cut_size = np.int(self.min_samples_split / 2) - 1

        splitted_sorted_y = sorted_y[cut_size:cut_size + sorted_y.size]

        r_border_ids = np.where(splitted_sorted_y[:-1] !=
                                splitted_sorted_y[1:])[0] + cut_size + 1

        if len(r_border_ids) == 0:
            return float('+inf'), None

        eq_el_count = r_border_ids - np.append([cut_size], r_border_ids[:-1])

        one_hot_code = np.zeros((r_border_ids.shape[0], max_class))
        one_hot_code[np.arange(r_border_ids.shape[0]),
                     sorted_y[r_border_ids - 1]] = 1

        class_increments = one_hot_code * eq_el_count.reshape(-1, 1)

        class_increments[0] = class_increments[0] + np.bincount(
            sorted_y[:cut_size],
            minlength=max_class
        )

        l_class_count = np.cumsum(class_increments, axis=0)

        r_class_count = np.bincount(sorted_y) - l_class_count

        l_sizes = r_border_ids.reshape(l_class_count.shape[0], 1)
        r_sizes = sorted_y.shape[0] - l_sizes

        gs = self.G_function(l_class_count, l_sizes, r_class_count, r_sizes)

        idx = np.argmin(gs)

        left_el_id = l_sizes[idx][0]
        return gs[idx], (sorted_x[left_el_id - 1] + sorted_x[left_el_id]) / 2.0

    def __fit_node(self, x, y, node_id, depth, pred_f=-1):

        if node_id == 0:
            self.tree.clear()

        if y.size == 0:
            return
        if np.unique(y).shape[0] == 1:
            self.__init_leaf(x, y, node_id)
            return

        if depth == self.max_depth or \
                np.max(np.bincount(y)) >= \
                np.int64(y.size * self.sufficient_share) or \
                y.size < self.min_samples_split:
            self.__init_leaf(x, y, node_id)
            return

        feature_ids = self.get_feature_ids(x.shape[1])

        classes_numb = np.max(y) + 1
        thresholds = np.zeros((2, x.shape[1]), dtype=np.float32)
        for i in range(x.shape[1]):

            if np.in1d(i, feature_ids):
                tmp, tmp_c = np.unique(x[:, i], return_counts=True)
                if tmp.shape[0] == 2:
                    thresholds[1, i] = (tmp[0] + tmp[1]) / 2
                    l_c = np.bincount(y[np.where(x[:, i] == tmp[0])],
                                      minlength=classes_numb
                                      )
                    r_c = np.bincount(y) - l_c
                    thresholds[0, i] = np.float(
                        self.G_function(l_c, tmp_c[0], r_c, tmp_c[1])
                    )
                else:
                    thresholds[:, i] = np.array(
                        self.__find_threshold(x[:, i], y))

        tmp = np.min(thresholds[0])
        tmp_ids = np.where(thresholds[0] == tmp)[0]
        feature_id = np.random.choice(tmp_ids)
        threshold = thresholds[1, feature_id]

        x_l, x_r, y_l, y_r = self.__div_samples(x, y, feature_id, threshold)

        if y_l.size == 0 or y_r.size == 0:
            self.__init_leaf(x, y, node_id)
            return
        self.__set_importance(y_l, y_r, feature_id)
        self.__init_none_leaf(node_id, feature_id, threshold)
        self.__fit_node(x_l, y_l, 2 * node_id + 1, depth + 1, feature_id)
        self.__fit_node(x_r, y_r, 2 * node_id + 2, depth + 1, feature_id)

    def __init_leaf(self, x, y, node_id):
        node = []
        node.append(self.LEAF_TYPE)
        tmp = np.bincount(y)
        tmp_probas = tmp / tmp.sum()
        node.append(np.argmax(tmp))
        node.append(tmp_probas)
        self.tree.update({node_id: node})

    def __init_none_leaf(self, node_id, feature_id, threshold):
        node = []
        node.append(self.NON_LEAF_TYPE)
        node.append(feature_id)
        node.append(threshold)
        self.tree.update({node_id: node})

    def __set_importance(self, y_l, y_r, feature_id):
        classes_numb = np.max(np.array([np.max(np.bincount(y_l)),
                                        np.max(np.bincount(y_r))])) + 1
        self.feature_importances_[feature_id] += \
            y_l.size / (y_l.size + y_r.size) * \
            (1.0 - self.G_function(np.bincount(y_l, minlength=classes_numb),
                                   y_l.size, y_r.size, y_r.size)
             )

    def fit(self, x, y):
        self.feature_importances_ = np.zeros((x.shape[1],), dtype=np.float32)
        self.num_class = np.unique(y).size
        self.__fit_node(x, y, 0, 0)

    def __predict_class(self, x, node_id):
        node = self.tree[node_id]
        if node[0] == self.__class__.NON_LEAF_TYPE:
            _, feature_id, threshold = node
            if x[feature_id] <= threshold:
                return self.__predict_class(x, 2 * node_id + 1)
            else:
                return self.__predict_class(x, 2 * node_id + 2)
        else:
            return node[1]

    def __predict_probs(self, x, node_id):
        node = self.tree[node_id]
        if node[0] == self.__class__.NON_LEAF_TYPE:
            _, feature_id, threshold = node
            if x[feature_id] <= threshold:
                return self.__predict_probs(x, 2 * node_id + 1)
            else:
                return self.__predict_probs(x, 2 * node_id + 2)
        else:
            return node[2]

    def predict(self, X):
        return np.array([self.__predict_class(x, 0) for x in X])

    def predict_probs(self, X):
        return np.array([self.__predict_probs(x, 0) for x in X])

    def fit_predict(self, x_train, y_train, predicted_x):
        self.fit(x_train, y_train)
        return self.predict(predicted_x)

    def score(self, x, y):
        pred_y = self.predict(x)
        return 1.0 - (np.bincount(np.abs(y - pred_y))[1:]).sum() / y.size


class MySGDClassifier:

    def __init__(self, C=0.5, alpha=0.01, max_epoch=5, batch_size=32):
        '''
        C - коэф. регуляризации
        alpha - скорость спуска
        max_epoch - максимальное количество эпох
        batch_size - максимальный размер мини батча
        '''

        self.C = C
        self.alpha = alpha
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y, count_losses=False):
        '''
        Обучение модели
        '''
        self.model = np.random.uniform(
            low=-0.1,
            high=0.1,
            size=(X.shape[1] + 1)
        )
        return self.partial_fit(X, y, count_losses)

    def partial_fit(self, X, y, count_losses=False):
        '''
        Метод дообучения модели на новых данных
        '''
        curr_epoch = 0

        losses = []

        while curr_epoch < self.max_epoch:
            indicies = np.arange(X.shape[0])
            np.random.shuffle(indicies)

            while indicies.size > 0:
                batch_indicies = np.random.choice(
                    indicies,
                    size=np.min((self.batch_size, indicies.size)),
                    replace=True
                )
                indicies = np.setdiff1d(indicies, batch_indicies)
                grad = self.__gradient(
                    X[batch_indicies],
                    y[batch_indicies]
                )
                self.model -= self.alpha / (curr_epoch + 1) * grad

                if count_losses:
                    losses.append(self.__loss(X, y))

            curr_epoch += 1

        if count_losses:
            return self, losses
        else:
            return self

    def predict(self, X):
        '''
        Возвращение метки класса
        '''
        predict_data = (X * self.model[1:]).sum(axis=1) + self.model[0]
        y_hat = np.zeros((X.shape[0],), dtype=np.float64)
        y_hat[predict_data > 0] = 1.0
        y_hat[predict_data < 0] = 0.0
        return y_hat

    def predict_proba(self, X):
        '''
        Возвращение вероятности каждого из классов
        '''
        predict_data = (X * self.model[1:]).sum(axis=1) + self.model[0]
        predict_data[predict_data > 0] /= predict_data.max() / 4
        predict_data[predict_data < 0] /= np.abs(predict_data.min()) / 4
        sigm = 1 / (1 + np.exp(-1 * predict_data))
        y_hat_proba = np.empty((X.shape[0], 2), dtype=np.float64)

        y_hat_proba[:, 1] = sigm
        y_hat_proba[:, 0] = 1 - sigm
        return y_hat_proba

    def __loss(self, X, y):
        tmp_y = y * 2 - 1
        margin = tmp_y * ((X * self.model[1:]).sum(axis=1) + self.model[0])
        margin[margin > 0] /= margin.max() / 4
        margin[margin < 0] /= np.abs(margin.min()) / 4
        res = (np.log(1 + np.exp((-1) * margin)).sum() / X.shape[0] -
               np.abs(self.model).sum() * self.C / X.shape[0]
               )
        return res

    def __gradient(self, X, y):
        tmp_y = y * 2 - 1
        margin = tmp_y * ((X * self.model[1:]).sum(axis=1) + self.model[0])
        margin[margin > 0] /= margin.max() / 4
        margin[margin < 0] /= np.abs(margin.min()) / 4
        mul = (tmp_y * (-1 / (1 + np.exp(margin)))).reshape(-1, 1) * X
        res = np.empty((X.shape[1] + 1,), dtype=np.float64)
        res[1:] = (mul.sum(axis=0) / X.shape[0] -
                   np.sign(self.model[1:]) * self.C / X.shape[0]
                   )
        res[0] = (tmp_y * (-1 / (1 + np.exp(margin)))).sum()
        return res


class MyKmeans:

    def __init__(self,
                 k=2,
                 metric='euclidean',
                 max_iter=1000,
                 random_state=None,
                 init='random',
                 eps=0.001
                 ):
        '''
        Инициализация метода
        :k - количество кластеров
        :metric - функция расстояния между объектами
        :max_iter - максиальное количество итераций
        :random_state - seed для инициализации генератора случайных чисел
        '''

        self.k = k
        self.random_state = random_state
        self.metric = metric
        self.max_iter = max_iter
        self.init = init
        self.metric = metric
        self.labels = None
        self.centroids = None
        self.inertia_ = None
        self.eps = eps

    def fit(self, X, count_inertia=False):
        '''
        Процедура обучения k-means
        '''

        # Инициализация генератора случайных чисел
        np.random.seed(self.random_state)

        # Массив с метками кластеров для каждого объекта из X
        self.labels = np.empty(X.shape[0])

        # Массив с центройдами кластеров
        self.centroids = np.empty((self.k, X.shape[1]))

        self._init_centroids(self.init, X)
        dist = np.empty((self.k, X.shape[0]))

        tmp_intervals = np.max(X, axis=0) - np.min(X, axis=0)

        for _ in range(self.max_iter):

            if self.metric == 'euclidean':
                for i in range(self.k):
                    dist[i] = self._euclidean(X, self.centroids[i])
            else:
                assert 'metric not Implemented'

            self.labels = np.argmin(dist, axis=0)
            dif = np.empty(self.k)
            for i in range(self.k):
                k_labels = (self.labels == i)
                tmp = np.mean(X[k_labels, :], axis=0)
                dif[i] = (np.abs(self.centroids[i] - tmp) /
                          tmp_intervals
                          ).sum()
                self.centroids[i] = tmp
            dif /= X.shape[1]
            if np.all(dif <= self.eps):
                break

        if count_inertia:
            self.inertia_ = 0
            for i in range(self.k):
                tmp = np.abs(X[self.labels == i] - self.centroids[i])
                self.inertia_ += (tmp * tmp).sum()

        return self

    def predict(self, X):
        '''
        Процедура предсказания кластера

        Возвращает метку ближайшего кластера для каждого объекта
        '''
        dist = np.empty((self.k, X.shape[0]))
        if self.metric == 'euclidean':
            for i in range(self.k):
                dist[i] = self._euclidean(X, self.centroids[i])
        else:
            assert 'metric not Implemeted'
        return np.argmin(dist, axis=0)

    def _init_centroids(self, method, X):

        if method == 'random':
            centroids_indicies = np.random.choice(np.arange(X.shape[0]),
                                                  size=self.k,
                                                  replace=False
                                                  )
            self.centroids = X[centroids_indicies]

        elif method == 'k-means++':

            self.centroids[0, :] = X[np.random.randint(low=0, high=X.shape[0])]
            dist = np.zeros((self.k, X.shape[0]))
            for i in range(1, self.k):
                for j in range(i):
                    dist[j] = self._euclidean(X, self.centroids[j])
                min_dist = np.min(dist[:i, :], axis=0)
                sum_ = min_dist.sum()
                centr_sum = np.random.uniform(0, sum_)
                sums = np.cumsum(min_dist)
                self.centroids[i, :] = X[sums > centr_sum][0]
        else:
            assert 'init not Implemented'

    @staticmethod
    def _euclidean(X, centroid):
        tmp = X - centroid
        return (tmp * tmp).sum(axis=1)


class MyMiniBatchKMeans(MyKmeans):
    def __init__(self,
                 k=2,
                 metric='euclidean',
                 max_iter=1000,
                 random_state=None,
                 init='random',
                 batch_size=0.4,
                 eps=1
                 ):
        super().__init__(k,
                         metric,
                         max_iter,
                         random_state,
                         init,
                         eps=eps
                         )
        self.batch_size = batch_size

    def fit(self, X, count_inertia=False):
        '''
        Процедура обучения k-means
        '''

        # Инициализация генератора случайных чисел
        np.random.seed(self.random_state)

        # Массив с метками кластеров для каждого объекта из X
        self.labels = np.zeros(X.shape[0])

        # Массив с центройдами кластеров
        self.centroids = np.empty((self.k, X.shape[1]))

        super()._init_centroids(self.init, X)

        batch_sz = int(X.shape[0] * self.batch_size)
        dist = np.empty((self.k, batch_sz))

        tmp_intervals = np.max(X, axis=0) - np.min(X, axis=0)
        tmp_counter = np.zeros(self.k)

        for j in range(self.max_iter):
            batch = np.random.randint(low=0, high=X.shape[0], size=batch_sz)
            if self.metric == 'euclidean':
                for i in range(self.k):
                    dist[i] = super()._euclidean(X[batch], self.centroids[i])
            else:
                assert 'metric not Implemented'

            tmp_labels = np.argmin(dist, axis=0)
            dif = np.zeros(self.k)

            for i in range(self.k):
                k_labels = (tmp_labels == i)
                if k_labels.sum() > 2:
                    tmp_counter[i] += 1
                    rate = 1 / tmp_counter[i]
                    tmp = (1 - rate) * self.centroids[i] +\
                        rate * np.mean(X[batch][k_labels], axis=0)
                    dif[i] = (np.abs(self.centroids[i] - tmp) / tmp_intervals).sum()
                    self.centroids[i] = tmp
            dif /= X.shape[1]
            if np.all(dif < self.eps):
                break

        dist = np.empty((self.k, X.shape[0]))
        for i in range(self.k):
            dist[i] = super()._euclidean(X, self.centroids[i])
        self.labels = np.argmin(dist, axis=0)

        if count_inertia:
            self.inertia_ = 0
            for i in range(self.k):
                tmp = np.abs(X[self.labels == i] - self.centroids[i])
                self.inertia_ += (tmp * tmp).sum()
        return self


class EMGaussianMixture:

    def __init__(self, k=3, max_iter=10, tol=0.001):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.expected_vs = None
        self.sigmas = None
        return

    def fit(self, x):

        expected_vs = self._init_centroids(x)
        self.expected_vs = np.copy(expected_vs)
        sigmas = [
                  np.matrix(np.diag(np.ones(x.shape[1])))
                  for _ in range(self.k)
                 ]
        weights = np.empty(self.k)
        weights.fill(1 / self.k)
        likelihoods = np.empty((self.k, x.shape[0]))

        for _ in range(self.max_iter):

            for i in range(self.k):
                tmp = np.matrix(x - expected_vs[i])
                likelihoods[i] = 1 / (np.pi ** (x.shape[1] / 2)) * \
                    1 / (np.linalg.det(sigmas[i]) ** (1 / 2)) * \
                    np.exp((-1 / 2) *
                           (np.array(tmp * np.linalg.inv(sigmas[i])) *
                           np.array(tmp)
                            ).sum(axis=1)
                           )

            tmp = np.array(likelihoods) * weights.reshape(-1, 1)

            mask = tmp.sum(axis=0) == 0
            if np.any(mask):
                dist = np.tile(expected_vs, (mask.sum(), 1)) - \
                       np.repeat(x[mask], expected_vs.shape[0], axis=0)
                dist = (dist * dist).sum(axis=1)
                tmp.T[mask] = (dist.reshape((-1, self.k), order='F'))

            gamma = tmp / tmp.sum(axis=0)
            weights = gamma.mean(axis=1)

            for i in range(self.k):
                tmp_sum = gamma[i].sum()
                tmp = x - expected_vs[i]
                tmp_sigm = np.matrix(np.empty((x.shape[1], x.shape[1])))
                for j in range(x.shape[1]):
                    tmp_sigm[j] = ((tmp[:, j].reshape(-1, 1) * tmp) *
                                   gamma[i].reshape(-1, 1)
                                   ).sum(axis=0) / tmp_sum
                if np.linalg.matrix_rank(tmp_sigm) == x.shape[1]:
                    sigmas[i] = tmp_sigm

                expected_vs[i] = (x * gamma[i].reshape(-1, 1)).sum(axis=0) / \
                    tmp_sum

            tmp_ = expected_vs - self.expected_vs
            if ((tmp_ * tmp_).sum(axis=1)).mean() < self.tol:
                break
            self.expected_vs = np.copy(expected_vs)

        self.weights = weights
        self.expected_vs = expected_vs
        self.sigmas = sigmas
        return self

    def predict(self, x):

        likelihoods = np.empty((self.k, x.shape[0]))
        for i in range(self.k):
            tmp = np.matrix(x - self.expected_vs[i])
            likelihoods[i] = 1 / (np.pi ** (x.shape[1] / 2)) * \
                1 / (np.linalg.det(self.sigmas[i]) ** (1 / 2)) * \
                np.exp((-1 / 2) *
                       (np.array(tmp * np.linalg.inv(self.sigmas[i])) *
                        np.array(tmp)
                        ).sum(axis=1)
                       )

        tmp = np.array(likelihoods) * self.weights.reshape(-1, 1)
        gamma = tmp / tmp.sum(axis=0)
        return np.argmax(gamma, axis=0)

    def fit_predict(self, x):
        self.fit(x)
        return self.predict(x)

    def _init_centroids(self, X):
        centroids = np.empty((self.k, X.shape[1]))
        centroids[0] = X[np.random.randint(low=0, high=X.shape[0])]
        dist = np.zeros((self.k, X.shape[0]))
        for i in range(1, self.k):
            for j in range(i):
                dist[j] = self._euclidean(X, centroids[j])
            min_dist = np.min(dist[:i, :], axis=0)
            sum_ = min_dist.sum()
            centr_sum = np.random.uniform(0, sum_)
            sums = np.cumsum(min_dist)
            centroids[i] = X[sums > centr_sum][0]
        return centroids

    @staticmethod
    def _euclidean(X, centroid):
        tmp = X - centroid
        return (tmp * tmp).sum(axis=1)

