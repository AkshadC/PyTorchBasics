import torch

print(torch.__version__)


def creating_tensors():
    scalar = torch.tensor(7)
    print(scalar)
    print()

    zero_tensor = torch.zeros([6, 9], dtype=torch.int32)
    print(zero_tensor)

def main():
    creating_tensors()


if __name__ == "__main__":
    main()
