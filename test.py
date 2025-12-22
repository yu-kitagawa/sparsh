import hydra
import torch
from omegaconf import DictConfig

from tactile_ssl.data.vision_based_interactive import DemoForceFieldData


@hydra.main(version_base="1.3", config_path="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = hydra.utils.instantiate(cfg.task)
    print("Testing")
    model.to(device)
    sensor_handler = DemoForceFieldData(
        config=cfg.data.dataset.config,
        digit_serial=None,
        gelsight_device_id=0,
    )
    sample = sensor_handler.get_model_inputs()
    x = sample["image"]
    x = x.unsqueeze(0).to(device)

    z = model.model_encoder(x)
    output = model.model_task.pooler(z).squeeze()
    print(output.shape)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()
