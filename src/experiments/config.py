# Maps each used forecast horizon to the optimal features found during feature selection
OPTIMAL_FEATURES = {
    1: {
        "station_features": [
            "wind_speed",
            "wind_direction_sin",
            "wind_direction_cos",
            "air_pressure",
            "air_temperature",
        ],
        "global_features": ["sin_1y", "cos_1y"],
    },
    4: {
        "station_features": [
            "wind_speed",
            "wind_direction_sin",
            "wind_direction_cos",
            "air_pressure",
            "air_temperature",
        ],
        "global_features": ["sin_1d", "cos_1d", "sin_1y", "cos_1y"],
    },
    8: {
        "station_features": [
            "wind_speed",
            "wind_direction_sin",
            "wind_direction_cos",
            "air_pressure",
            "relative_humidity",
        ],
        "global_features": ["sin_1d", "cos_1d", "sin_1y", "cos_1y"],
    },
}

# Exact model architectures employed in the thesis
MODEL_ARCHS = {
    "mlp": {
        "num_hidden_dense_layers": 2,
        "dense_hidden_size": 64,
    },
    "lstm": {
        "num_hidden_lstm_layers": 1,
        "lstm_hidden_size": 128,
        "num_hidden_dense_layers": 2,
        "dense_hidden_size": 64,
    },
    "gcn": {
        "hidden_channels": 128,
        "num_hidden_gnn_layers": 2,
        "use_residual": True,
        "num_hidden_dense_layers": 2,
        "dense_hidden_size": 32,
    },
    "gcn_lstm": {
        "hidden_channels": 128,
        "num_hidden_gnn_layers": 2,
        "use_residual": True,
        "num_hidden_lstm_layers": 1,
        "lstm_hidden_size": 256,
        "num_hidden_dense_layers": 0
    },
}
