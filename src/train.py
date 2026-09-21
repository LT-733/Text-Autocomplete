import encoder
import model
import torch
import torch.nn as nn
import torch.optim as optim
import jaxtyping
from jaxtyping import Int
import random
import sampler
from typing import Optional

def main(complete: Optional[model.Model]) -> None:
    device: torch.Device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")

    if complete is not None:
        print("loaded")
    diffusion_model: model.Model = model.Model().to(device) if complete is None else complete.to(device=device)

    with open("../data/input.txt") as f:
        lines: list[str] = f.readlines()

    line: str = "\n".join(lines)

    corpus: torch.Tensor = torch.tensor(diffusion_model.Encoder.encode(line), dtype=torch.long, device=device)
    sequence_len: int = 128
    encoded_len: int = corpus.shape[0]
    batch_size: int = 64
    # print(encoded_len)
    print(diffusion_model.Encoder.lookuptable())


    
    epochs: int = 1000
    sequence: torch.Tensor = torch.arange(sequence_len, device=device)
    lowest_loss = 1.0000

    # takes None, returns prepared training tensors of shape (batch_size, sequence_len), which is data randomly pulled out of the encoded text by
    # the model's own encoder
    def build_batch() -> torch.Tensor:
        start_tensor: torch.Tensor = torch.randint(low=0, high=(encoded_len - sequence_len), size=(batch_size, ), requires_grad=False, device=device)
        # returns a part of corpus, indexed by a tensor of shape (batch_size, sequence_len), or just (64, 128)
        return corpus[start_tensor[:, None] + sequence[None, :]]
    
    def add_noise(naive_tensor: Int[torch.Tensor, "batch_size sequence_len"], prob: float) -> tuple[torch.Tensor, torch.Tensor]:
        # this basically gives you zeros and ones, effectively masking out some of the tensors, once you apply this mask onto your naive input.
        modification: Int[torch.Tensor, "batch_size sequence_len"] = (torch.rand((naive_tensor.shape), device=device) < prob).long()
        # by applying this mask, you added noise to your input tensor for your model to predict
        masked_input: Int[torch.Tensor, "batch_size sequence_len"] = naive_tensor * (1 - modification)
        return modification, masked_input

    optimizer = optim.Adam(diffusion_model.parameters(), lr=1e-8)

    for i in range(0, epochs+1):
        train_input: Int[torch.Tensor, "batch_size sequence_len"] = build_batch()
        mask_probability = random.uniform(0, 1)
        mask: Int[torch.Tensor, "batch_size sequence_len"]
        mask, noised_input = add_noise(naive_tensor=train_input, prob=mask_probability)
        mask = mask.to(device=device)

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            predictions: torch.Tensor = diffusion_model.forward(noised_input, probabilities=torch.tensor([mask_probability], device=device))

            per_token_loss = nn.functional.cross_entropy(
                input=predictions.reshape(-1, predictions.shape[-1]),
                target=train_input.reshape(-1),
                reduction="none"
            ).reshape(batch_size, sequence_len)

            loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1)

        if loss < lowest_loss:
            print("saving...")
            torch.save(diffusion_model.state_dict(), "../saved_weights/temp_model.pt")
            lowest_loss = loss

        if i % 10 == 0:
            print(f"step: {i}, loss: {loss.item(): .4f}")
        if i % 1000 == 0 and i != 0:
            with open("../saved_weights/log.txt", "a") as o:
                progress: str = sampler.sample(model=diffusion_model, query="To be, ", length=64, device=device)#type: ignore
                o.write(progress)
                o.write("\n")
            # o.close()

        with torch.autograd.set_detect_anomaly(True, check_nan=False):
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()

if __name__ == "__main__":
    try:
        tst_model = model.Model().to(device="mps" if torch.mps.is_available() else "cuda")
        tst_model.load_state_dict(torch.load("../saved_weights/temp_model.pt", map_location=torch.device("mps" if torch.mps.is_available() else "cuda")))
        main(tst_model)
    except Exception as e:
        main(None)







    

