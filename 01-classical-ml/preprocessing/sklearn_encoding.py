from numpy import asarray
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

data =asarray([["red"],
              ["green"],
              ["blue"],
              ["purple"],
              ["green"]])
print(data)
# ordinal = OrdinalEncoder()
# out = ordinal.fit_transform(data)
# print(out)
encoder = OneHotEncoder(sparse_output=False)
onehot = encoder.fit_transform(data)
print(onehot)

