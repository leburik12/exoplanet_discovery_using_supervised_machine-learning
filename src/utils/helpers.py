import numpy as np
from sklearn.preprocessing import PowerTransformer
from scipy import stats

def suggest_transformation(data):
    if all(data > 0):
        transformer = PowerTransformer(method='yeo-johnson', standardize=True)
        transformed = transformer.fit_transform(data.values.reshape(-1, 1)).flatten()
        pval = stats.normaltest(transformed).pvalue
        return pval < 0.05, pval
    return False, np.nan
