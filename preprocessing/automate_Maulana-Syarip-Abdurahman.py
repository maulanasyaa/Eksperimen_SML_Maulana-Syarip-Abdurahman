import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


def preprocess_data(train_path, test_path=None):
    """
    Preprocessing Function

    Args:
        train_path (str): Path to train data CSV
        test_path (str, optional): Path to test data CSV. Defaults to None.

    Returns:
        tuple: (train_processed, test_processed, preprocessor, label_encoder)
                - train_processed: Processed train data DataFrame.
                - test_processed: Processed test data DataFrame.
                - preprocessor: ColumnTransformer Object.
                - label_encoder: LabelEncoder Object.
    """

    # data loading
    print("Loading datasets...")
    df = pd.read_csv(train_path)

    # missing value
    print("Handling missing values...")
    df = df.dropna()

    # separate feature and target
    target_col = "Personality"
    id_col_name = "id"

    X = df.drop(columns=[id_col_name, target_col], axis=1)
    y = df[target_col]
    train_id = df[id_col_name]

    # separate features
    numerical_features = X.select_dtypes(include=np.number).columns
    categorical_features = X.select_dtypes(include=np.object_).columns

    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    # preprocessing pipeline
    print("Create preprocessing pipeline...")
    scaler = StandardScaler()
    encoder = OneHotEncoder(handle_unknown="ignore")

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", scaler)]
    )

    categoric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numerical_features),
            ("categoric", categoric_transformer, categorical_features),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    # feature preprocessing
    print("Transform...")
    preprocessor.fit(X)
    X_preprocessed = preprocessor.transform(X)
    X_preprocessed = pd.DataFrame(
        X_preprocessed, columns=preprocessor.get_feature_names_out()
    )

    # target preprocessing
    le = LabelEncoder()
    y_preprocessed = le.fit_transform(y)
    y_preprocessed = pd.DataFrame(y_preprocessed, columns=[target_col])

    # combine preprocessed features and target
    train_processed = pd.concat(
        [train_id.reset_index(drop=True), X_preprocessed, y_preprocessed], axis=1
    )
    train_processed[id_col_name] = train_processed[id_col_name].astype(int)

    # test data preprocessing
    test_processed = None
    if test_path:
        print("Processing test data...")
        # load data
        df_test = pd.read_csv(test_path)

        # missing values
        df_test = df_test.dropna()

        test_id = df_test[id_col_name]
        X_test = df_test.drop(columns=[id_col_name])

        # transform
        X_test_preprocessed = preprocessor.transform(X_test)
        X_test_preprocessed = pd.DataFrame(
            X_test_preprocessed, columns=preprocessor.get_feature_names_out()
        )

        # combine features with id
        test_processed = pd.concat(
            [test_id.reset_index(drop=True), X_test_preprocessed], axis=1
        )
        test_processed[id_col_name] = test_processed[id_col_name].astype(int)

    print("Preprocessing complete.")
    return train_processed, test_processed, preprocessor, le


if __name__ == "__main__":
    TRAIN_PATH = "../predict_the_introverts_from_the_extroverts_raw/train.csv"
    TEST_PATH = "../predict_the_introverts_from_the_extroverts_raw/test.csv"
    OUTPUT_TRAIN = "predict_the_introverts_from_the_extroverts_preprocessing/train_preprocessing.csv"
    OUTPUT_TEST = "predict_the_introverts_from_the_extroverts_preprocessing/test_preprocessing.csv"

    try:
        train_data, test_data, _, _ = preprocess_data(TRAIN_PATH, TEST_PATH)

        # convert to csv
        train_data.to_csv(OUTPUT_TRAIN, index=False)
        print(f"Train data saved on: {OUTPUT_TRAIN}")

        if test_data is not None:
            test_data.to_csv(OUTPUT_TEST, index=False)
            print(f"Test data saved on: {OUTPUT_TEST}")
    except FileNotFoundError as e:
        print(f"Error: File not found.: {e}")
    except Exception as e:
        print(f"Error: {e}")
