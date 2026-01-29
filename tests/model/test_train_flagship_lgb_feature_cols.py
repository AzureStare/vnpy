import polars as pl


def test_select_feature_columns_excludes_forward_looking_cols() -> None:
    from flagship.model.train_flagship_lgb import select_feature_columns

    df = pl.DataFrame(
        {
            "datetime": ["2025-01-02", "2025-01-02"],
            "vt_symbol": ["AAA.NASDAQ", "BBB.NASDAQ"],
            "rank_5d": [1, 2],
            "label": [0.1, -0.2],
            "ret_5d": [0.3, -0.1],
            "label_excess_5d": [0.25, -0.05],
            "some_feature": [10.0, 20.0],
        }
    )

    feature_cols = select_feature_columns(df, label_col="rank_5d")
    assert "some_feature" in feature_cols
    assert "ret_5d" not in feature_cols
    assert "label_excess_5d" not in feature_cols

    feature_cols_2 = select_feature_columns(df, label_col="label_excess_5d")
    assert "some_feature" in feature_cols_2
    assert "ret_5d" not in feature_cols_2
    assert "label_excess_5d" not in feature_cols_2

