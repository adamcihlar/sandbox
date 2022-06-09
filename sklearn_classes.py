
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

def cat_num_colnames(data):
    cat_columns = data.select_dtypes(include=['category', 'object']).columns
    num_columns = data.select_dtypes(exclude=['category', 'object']).columns
    return cat_columns, num_columns

class Simple_Imputing(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        cat_cols, num_cols = cat_num_colnames(data)

        modus = SimpleImputer(strategy="most_frequent")
        mean = SimpleImputer(strategy="mean")
        data = pd.DataFrame(pd.concat([pd.DataFrame(modus.fit_transform(data[cat_cols]), columns = cat_cols),
                                pd.DataFrame(mean.fit_transform(data[num_cols]), columns = num_cols)],
                               axis=1))
        data[cat_cols] = data[cat_cols].astype('category')

        return data

si = Simple_Imputing()

train_si = si.transform(train_feat)




from sklearn.compose import ColumnTransformer, make_column_transformer, make_column_selector
from sklearn.impute import KNNImputer
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

# knn imputer knows only euclidian dist - we need to define our own
def manhattan_dist(x, y, missing_values=np.nan):
    missings_mask = np.array(~np.isnan(x) & ~np.isnan(y))
    weight = len(x) / sum(missings_mask)
    x = x[missings_mask]
    y = y[missings_mask]
    return np.mean(np.abs(x - y) * weight)

def ReverseMinMaxScaler(data, mins, ranges):
    return data * ranges + mins

class Imputing(BaseEstimator, TransformerMixin):
    def __init__(self,n_neighb=None):
            self.n_neighb = n_neighb

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        # save names of cat cols for further work
        cat_cols, num_cols = cat_num_colnames(data)

        if self.n_neighb==None:
            modus = SimpleImputer(strategy="most_frequent")
            mean = SimpleImputer(strategy="mean")

            data = pd.DataFrame(pd.concat([pd.DataFrame(modus.fit_transform(data[cat_cols]), columns = cat_cols),
                                pd.DataFrame(mean.fit_transform(data[num_cols]), columns = num_cols)],
                               axis=1)
                    )
            data[cat_cols] = data[cat_cols].astype('category')

        else:
            # encode categoricals to numericals to be able to use KNN imputer
            data_cat = data.select_dtypes(include='category').apply(lambda series: pd.Series(
                LabelEncoder().fit_transform(series[series.notnull()]),
                index=series[series.notnull()].index
            ))

            # concat labeled categoricals back to the other explanatory vars to have everything for KNN imputer
            data = pd.concat([data_cat, data.select_dtypes(exclude='category')], axis=1)

            # save statistics to get original scale after imputation
            mins = np.min(data)
            ranges = np.max(data) - np.min(data)

            # vars are numericals, more neighbors will give real numbers - take as prob and round (for categoricals)
            imputer = KNNImputer(n_neighbors=self.n_neighb, metric=manhattan_dist)

            pip = make_pipeline(MinMaxScaler(), imputer)
            data = pd.DataFrame(pip.fit_transform(data), columns=data.columns)

            # get original scales
            data = ReverseMinMaxScaler(data, mins, ranges)

            # round categorical vars
            data[cat_cols] = data[cat_cols].round(decimals=0).astype('category')

        return data

knn_imp = Imputing(n_neighb=5)

train_knn = knn_imp.transform(train_feat)





class OneHot_Enc(BaseEstimator, TransformerMixin):
    def __init__(self,drop=None):
        # drop either 'first' (for model where perfect collinearity is a trouble - logit) or None
        self.drop = drop

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        cat_cols, num_cols = cat_num_colnames(data)

        # multiple pipelines to be able to retrieve the variable names later
        onehot = Pipeline([('onehot', OneHotEncoder(drop=self.drop))])
        prep = ColumnTransformer([('prep', onehot, cat_cols)],
                                remainder='passthrough')
        pip = Pipeline([('pip', prep)])

        pip.fit(data)
        # retrieve onehotencoded names and append the continuous
        cat_cols = pip.named_steps['pip'].transformers_[0][1].named_steps['onehot'].get_feature_names(cat_cols)
        colnames = np.append(cat_cols, num_cols)

        data = pd.DataFrame(pip.transform(data), columns=colnames)
        data[cat_cols] = data[cat_cols].astype('category')

        return data

ohc = OneHot_Enc()

train_knn = ohc.transform(train_knn)





from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Scaler(BaseEstimator, TransformerMixin):
    # std: {None, 'std', 'minmax'}
    def __init__(self, std=None):
        self.std = std

    def fit(self, data, y=None):
        return self

    def transform(self, data, y=None):
        cat_cols, num_cols = cat_num_colnames(data)
        if self.std == None:
            data = data.apply(pd.to_numeric)
            return data
        elif self.std == 'std':
            scaler = StandardScaler()
        elif self.std == 'minmax':
            scaler = MinMaxScaler()
        else:
            return data
        data_std = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        #data_std[cat_cols] = data_std[cat_cols] / 2 # the distance is computed twice for 'one difference' in cat vars
        return data_std

scale = Scaler(std='minmax')
train_knn_std = scale.transform(train_knn)




import statsmodels.api as sm

class tDistancing(BaseEstimator, TransformerMixin):
    def __init__(self, do=True):
        self.do=do

    def fit(self, data, y):
        logit = sm.Logit(y.values.reshape(-1,1), data).fit(maxiter = 500, disp=0) #.values.reshape(-1,1)
        t_vals = pd.DataFrame((logit.tvalues)).reset_index(level=0) #np.abs - no absolute value to keep distncs from t-vals

        cols = pd.DataFrame(data.columns)

        t_vals = pd.merge(cols, t_vals, how='left', left_on=0, right_on='index')['0_y']
        self.t_vals = t_vals
        return self

    def transform(self, data):
        if self.do:
            data = data.multiply(self.t_vals.T.squeeze().values, axis='columns')

            data['Dependents'] = np.sum(data[list(data.filter(regex='Dependents_'))], axis=1)
            data['Property_Area'] = np.sum(data[list(data.filter(regex='Property_Area_'))], axis=1)

            data = data[data.columns.drop(list(data.filter(regex='Dependents_')))]
            data = data[data.columns.drop(list(data.filter(regex='Property_Area_')))]
        return data
