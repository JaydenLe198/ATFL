"""Server application definition for ATFL."""

from __future__ import annotations

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from fl.task import build_model, get_weights
from fl.utils.config import load_config


def server_fn(context: Context) -> ServerAppComponents:
    cfg = load_config(context.run_config.get("config_path"))
    run_cfg = context.run_config

    num_rounds = int(run_cfg.get("num-server-rounds", cfg.get("simulation", {}).get("total_rounds", 1)))
    fraction_fit = float(run_cfg.get("fraction-fit", cfg.get("simulation", {}).get("fraction_fit", 1.0)))

    model_cfg = cfg.get("model", {})
    seq_len = int(model_cfg.get("seq_len", 100))
    model_kwargs = {
        "win_size": seq_len,
        "enc_in": int(model_cfg.get("enc_in", 25)),
        "c_out": int(model_cfg.get("c_out", 25)),
        "d_model": int(model_cfg.get("d_model", 512)),
        "n_heads": int(model_cfg.get("n_heads", 8)),
        "e_layers": int(model_cfg.get("e_layers", 3)),
    }

    ndarrays = get_weights(build_model(model_kwargs))
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
