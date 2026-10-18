import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

def demonstrate_encoding():
    # Sample categorical data
    data = np.asarray([["red"],
                      ["green"],
                      ["blue"],
                      ["purple"],
                      ["green"]])

    print("Original Categorical Data:")
    print(data)

    # 1. Ordinal Encoding (Maps categories to integers: 0, 1, 2, ...)
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoded = ordinal_encoder.fit_transform(data)
    print("\n--- Ordinal Encoding ---")
    print("Categories:", ordinal_encoder.categories_[0])
    print("Encoded values:\n", ordinal_encoded)

    # 2. One-Hot Encoding (Creates binary columns for each category)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_encoded = onehot_encoder.fit_transform(data)
    print("\n--- One-Hot Encoding ---")
    print("Categories:", onehot_encoder.categories_[0])
    print("One-Hot Matrix:\n", onehot_encoded)

if __name__ == "__main__":
    demonstrate_encoding()
