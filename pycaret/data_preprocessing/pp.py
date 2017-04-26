from sklearn.preprocessing import StandardScaler
from sklearn,decomposition import PCA
import pandas as pd

'''
desired behaviour:
list of strings in (e.g. 'center_scale', 'pca')

list of preprocessing classes created corresponding to strings

then called to apply to data frame
the attribute updated

pca either keeps enough features to get 95% of variance, or set n


'''

class PP():

  def __init__(self, pp_list, pca_comp = None, pca_thresh = 0.95):

    self.pp_list = pp_list
    self.pp_classes = []


    if pca_comp is not None & pca_thresh is not None:
      raise ValueError('choose number of components or threshold')

    self.pca_thresh = pca_thresh

    for _ in pp_list:
      if _ == 'cs':
        self.pp_classes.append(StandardScaler())

      if _ == 'pca':
        # note: if none then retains all.
        self.pp_classes.append(PCA(n_components = pca_comp))


  def __call__(self, df):
    # fit preprocessing and transform data in order
    # as use sklearn classes, can take advantage that
    # API is consistent

    colnames = [n for n in df.columns]

    print('**')
    print(df)
    for _ in self.pp_classes:
      df = _.fit_transform(df)

    # if
    if 'pca' in self.pp_list:
      if self.pca_thresh is not None:
        cumsum = 0.
        count = 0
        for i, in df.shape[1]:
          cumsum += df[i,i]
          count += 1
          if cumsum >= self.pca_thresh:
            break

        df = df[:, count]
        colnames = ['comp_'+str(i+1) for i in count]
        df = pd.DataFrame(df, columns = colnames)
        return df
      else:
        colnames = ['comp_'+str(i+1) for i in range(df.shape[1])]
        df = pd.DataFrame(df, columns = colnames)
        return df


    # IF NO PCA JUST ADD COLNAMES BACK
    df = pd.DataFrame(df, columns = colnames)

    return df
