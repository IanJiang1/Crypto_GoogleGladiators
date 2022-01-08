
from itertools import product
from hmmlearn import hmm
from sklearn.base import BaseEstimator
from pycaret.classification import *


class HMMClassifier(BaseEstimator):
    def __init__(self, n_components, mode="convert"):
      self.n_components = n_components
      self.mode = mode

    def fit(self, X, y, epsilon=0.1):
      if self.mode == "all":
        X_train = np.hstack([np.array(X), np.array(y).reshape(-1, 1)])
      elif self.mode == "target":
        X_train = np.array(y).reshape(-1, 1)
      elif self.mode == "cat":
        X_train = np.hstack([np.array(X.select_dtypes(exclude=["number"])), np.array(y.astype(str)).reshape(-1, 1)])
      elif self.mode == "convert":
        X_train = np.hstack([np.array(X.astype(str)), np.array(y).reshape(-1, 1)])

      # print(X_train[:10])
      print(X.head())
      print("X.nunique(): {}".format(X.nunique()))
      print("y.nunique: {}".format(y.nunique()))

      # Needa convert to series first!
      if len(np.squeeze(X_train).shape) > 1:
        test_values = [set(x) for x in X_train.T]
        X_train = pd.Series(map(tuple, X_train))

        prods = set(product(*test_values))
        observed_combos = set(X_train.unique())
        unobserved_combos = prods - observed_combos
        self.label_map = {**dict(zip(list(unobserved_combos), np.arange(len(unobserved_combos)))),
                          **dict(zip(list(observed_combos), np.arange(len(unobserved_combos), len(unobserved_combos) + len(observed_combos))))}

        self.inv_label_map = {v:k for k, v in self.label_map.items()}
        X_train = X_train.map(self.label_map).values.reshape(-1, 1)

      self.remodel = hmm.MultinomialHMM(n_components=self.n_components, n_iter=100)
      self.remodel.fit(X_train)
      self.transmat = self.remodel.transmat_
      self.Z = self.remodel.predict(X_train)
      self.curr_state = self.Z[-1]
      self.emissionprob = self.remodel.emissionprob_
      self.emissionprob += epsilon / self.emissionprob.shape[1]
      self.emissionprob /= self.emissionprob.sum(axis=1, keepdims=True)
      self.state_prob = pd.DataFrame(self.emissionprob).rename(self.inv_label_map, axis=1)
      return self

    def predict_viterbi(self, X):
      def get_relevant_cols(state_prob, partial_state):
        relevant_cols = [col for col in state_prob if col[:len(partial_state)] == partial_state]
        other_cols = list(set(state_prob.columns) - set(relevant_cols))
        return relevant_cols, other_cols
      
      def get_state_likelihood(state_prob, partial_state):
        relevant_cols, other_cols = get_relevant_cols(state_prob, partial_state)
        result = state_prob[relevant_cols].sum(axis=1)
        return np.squeeze(result.sort_index().values)
      
      # Restricting to categorical values
      if self.mode == "all":
        pass
      elif self.mode == "cat":
        X = X.select_dtypes(exclude=["number"])

      # Forward pass
      T1 = [np.squeeze(get_state_likelihood(self.state_prob, tuple(X.iloc[0].values)))]
      T2 = [np.squeeze(np.zeros(self.n_components))]
      count = 0
      for x in X.iloc[1:].iterrows():
        partial_probs = get_state_likelihood(self.state_prob, tuple(x[1].values))
        test_mat = (T1[count].reshape(-1, 1)*self.transmat)*partial_probs
        T1_row = np.squeeze(test_mat.max(axis=0))
        T2_row = np.squeeze(test_mat.argmax(axis=0))
        T1_row /= T1_row.sum()
        T1.append(T1_row)
        T2.append(T2_row)
        count += 1
      
      # Backward pass
      T1 = np.array(T1)
      T2 = np.array(T2)
      T1 /= T1.sum(axis=1).reshape(-1, 1)
      opt_last_state = T1[-1, :].argmax()
      pred_path = [opt_last_state]
      for j in range(X.shape[0] - 1):
        opt_state = T2[X.shape[0] - j - 1, int(pred_path[j])]
        pred_path.append(opt_state)
      pred_path = np.array(pred_path[::-1])
      
      # Assignment
      final_result = []
      final_probs = []
      for x, y in zip(X.iterrows(), pred_path):
        try:
          if self.mode == "convert":
            relevant_cols, _ = get_relevant_cols(self.state_prob, tuple(x[1].values.astype(str)))
          else:
            relevant_cols, _ = get_relevant_cols(self.state_prob, tuple(x[1].values))
          test_mat = self.state_prob[relevant_cols].loc[y]
          test_y = test_mat.index.map(lambda z: z[-1]).tolist()
          final_probs.append(test_mat.groupby(test_y).sum().sort_index().values)
          final_result.append(test_mat.idxmax()[-1])
        except:
          print(x)
          raise ValueError("Something's wrong, hmmmmm....")
      return T1, T2, pred_path, np.array(final_result), np.array(final_probs)
    
    def predict(self, X):
      _, _, _, y, _ = self.predict_viterbi(X)
      return y
    
    def predict_proba(self, X):
      _, _, _, _, y = self.predict_viterbi(X)
      return y/y.sum(axis=1).reshape(-1, 1)