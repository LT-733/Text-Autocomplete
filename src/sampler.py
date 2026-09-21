import torch
import torch.nn as nn
import encoder, train, model
import jaxtyping, random
import argparse

# Does autocomplete
def sample(model: model.Model, query: str, length: int, device: torch.device) -> str:
    tokens = model.Encoder.encode(query)
    raw_output: jaxtyping.Int[torch.Tensor, ""] = torch.zeros(length, dtype=torch.long, device=device)
    raw_output[:len(tokens)] = torch.tensor(tokens, device=device)
    fixed: torch.Tensor = (raw_output != 0)

    with torch.no_grad():
        for i in range(20):
            mask_prob: jaxtyping.Int[torch.Tensor, "1"] = torch.tensor([1.0 - i / 20], device=device)
            output = model.forward(x=raw_output, probabilities=mask_prob)
            mapped_output = torch.softmax(output, dim=-1)
            became_masked_pos: torch.Tensor = torch.logical_and((raw_output == 0), torch.logical_not(fixed))

            if not became_masked_pos.any():
                break
            for pos in became_masked_pos.nonzero():
                if random.random() < (1 / (20 - i)):
                    raw_output[pos] = torch.multinomial(mapped_output[pos], 1)

    final: str = ''.join(model.Encoder.decode(raw_output.tolist()))
    print(final)
    return final

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="../saved_weights/temp_model.pt")
    p.add_argument("--query", default="To be, ")
    p.add_argument("--length", default=64)

    device:torch.Device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")
    args = p.parse_args()
    test_model: model.Model = model.Model().to(device=device)
    test_model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    test_model.eval()

    out = sample(model=test_model, query=args.query, length=args.length, device=device) #type: ignore

if __name__ == "__main__":
    main()
