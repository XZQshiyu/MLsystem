from sklearn.datasets import load_wine
import numpy as np 
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors


def get_kernel_function(kernel:str):
    # TODO: implement different kernel functions 
    if kernel == "linear":
        return lambda x, y: np.dot(x, y.T)
    elif kernel == "poly":
        return lambda x, y, p = 3: (np.dot(x, y.T) + 1) ** p
    elif kernel == "rbf":
        return lambda x, y, sigma=5.0: np.exp(-np.linalg.norm(x[:, np.newaxis] - y[np.newaxis, :], axis=2) ** 2  / (2 * sigma ** 2))
    else:
        raise ValueError(f"Kernel {kernel} is not supported.")

class PCA:
    def __init__(self, n_components:int=2, kernel:str="rbf") -> None:
        # 主成分数量
        self.n_components = n_components
        # 核函数
        self.kernel_f = get_kernel_function(kernel)
        # kernel function type
        self.kernel = kernel
        # 尚未拟合的数据
        self.X_fit = None
        # 主成分的特征向量
        self.alpha = None
        # 主成分的特征值
        self.lambdas = None
        # ...

    def fit(self, X:np.ndarray):
        # X: [n_samples, n_features]
        # TODO: implement PCA algorithm
        self.X_fit = X
        K = self.kernel_f(X, X)
        n = X.shape[0]

        # centering the kernel matrix
        one_n = np.ones((n, n)) / n
        K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

        # eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(K)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        self.lambdas = eigvals[:self.n_components]
        self.alpha = eigvecs[:, :self.n_components]

    def transform(self, X:np.ndarray):
        # X: [n_samples, n_features]
        # X_reduced = np.zeros((X.shape[0], self.n_components))
        # TODO: transform the data to low dimension
        K = self.kernel_f(X, self.X_fit)
        return K.dot(self.alpha) / np.sqrt(self.lambdas * X.shape[0])

class KMeans:
    def __init__(self, n_clusters:int=3, max_iter:int=10) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = None
        self.labels = None

    # Randomly initialize the centers
    def initialize_centers(self, points):
        # points: (n_samples, n_dims,)
        n, d = points.shape

        self.centers = np.zeros((self.n_clusters, d))
        for k in range(self.n_clusters):
            # use more random points to initialize centers, make kmeans more stable
            random_index = np.random.choice(n, size=10, replace=False)
            self.centers[k] = points[random_index].mean(axis=0)
        
        return self.centers
    
    # Assign each point to the closest center
    def assign_points(self, points):
        # points: (n_samples, n_dims,)
        # return labels: (n_samples, )
        n_samples, n_dims = points.shape
        self.labels = np.zeros(n_samples)
        # TODO: Compute the distance between each point and each center
        # and Assign each point to the closest center
        for i in range(n_samples):
            distances = np.linalg.norm(points[i] - self.centers, axis=1)
            self.labels[i] = np.argmin(distances)
    
        return self.labels

    # Update the centers based on the new assignment of points
    def update_centers(self, points):
        # points: (n_samples, n_dims,)
        # TODO: Update the centers based on the new assignment of points
        for k in range(self.n_clusters):
            cluster_points = points[self.labels == k]
            if len(cluster_points) > 0:
                self.centers[k] = cluster_points.mean(axis=0)

    # k-means clustering
    def fit(self, points):
        # points: (n_samples, n_dims,)
        # TODO: Implement k-means clustering
        self.initialize_centers(points)
        for _ in range(self.max_iter):
            old_centers = self.centers.copy()
            self.assign_points(points)
            self.update_centers(points)
            if np.all(old_centers == self.centers):
                break

    # Predict the closest cluster each sample in X belongs to
    def predict(self, points):
        # points: (n_samples, n_dims,)
        # return labels: (n_samples, )
        return self.assign_points(points)
    
def load_data():
    words = [
        'computer', 'laptop', 'minicomputers', 'PC', 'software', 'Macbook',
        'king', 'queen', 'monarch','prince', 'ruler','princes', 'kingdom', 'royal',
        'man', 'woman', 'boy', 'teenager', 'girl', 'robber','guy','person','gentleman',
        'banana', 'pineapple','mango','papaya','coconut','potato','melon',
        'shanghai','HongKong','chinese','Xiamen','beijing','Guilin',
        'disease', 'infection', 'cancer', 'illness', 
        'twitter', 'facebook', 'chat', 'hashtag', 'link', 'internet',
    ]
    w2v = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary = True)
    vectors = []
    for w in words:
        vectors.append(w2v[w].reshape(1, 300))
    vectors = np.concatenate(vectors, axis=0)
    return words, vectors

if __name__=='__main__':
    words, data = load_data()
    pca = PCA(n_components=2)
    pca.fit(data)
    data_pca = pca.transform(data)

    kmeans = KMeans(n_clusters=7)
    kmeans.fit(data_pca)
    clusters = kmeans.predict(data_pca)

    # plot the data
    
    plt.figure()
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters)
    for i in range(len(words)):
        plt.annotate(words[i], data_pca[i, :]) 
    plt.title("PB21111653")
    plt.savefig("PCA_KMeans.png")
    plt.show()