import torch

print(f"Current Torch Version: {torch.__version__}, \n {torch.cuda.get_device_name(torch.cuda.current_device())}")
device = "cuda" if torch.cuda.is_available() else "cpu"

RANDOM_SEED = 69
torch.manual_seed(RANDOM_SEED)


def creating_tensors():
    scalar = torch.tensor(7)
    print(scalar)
    print()

    zero_tensor = torch.zeros([6, 9], dtype=torch.int32)
    print(zero_tensor)

    print("Scalar Dimension: ", scalar.ndim)
    print("Scalar Item: ", scalar.item())
    print("Size zero tensor: ", zero_tensor.shape)
    print("Ndim zero tensor: ", zero_tensor.ndim)

    TENSOR = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    print(f"Tensor: {TENSOR}, Shape: {TENSOR.shape}, Ndim: {TENSOR.ndim}")


def random_tensors():
    random_tensor = torch.randn([3, 3, 3], dtype=torch.float32, device="cuda")
    print(f"Random Tensor : {random_tensor}, \n Shape: {random_tensor.shape}, \n Ndim: {random_tensor.ndim}")

    random_tensor_image = torch.rand(size=(224, 224, 3))
    print(f" Shape: {random_tensor_image.shape}, \n Ndim: {random_tensor_image.ndim}")


def range_of_tensors():
    range_tensor = torch.arange(69, 101, step=4, device="cuda")
    print(f"Range Tensor: {range_tensor}, {range_tensor.dtype}, {range_tensor.device}")
    print(f"Multiply with 10: {torch.mul(range_tensor, 10)}")


def modifying_tensors():
    x = torch.arange(10, dtype=torch.float32)
    print(f"{x}, shape: {x.shape}")
    x_reshaped = x.reshape(5, 2)
    print(f"{x_reshaped}, shape: {x_reshaped.shape}")

    x_stacked = torch.stack([x, x, x, x], dim= 0)
    print(x_stacked)

    y = torch.arange(1, 10).reshape(1, 3, 3)
    print(y[0])

    print(y[:,1, 1])
    x = x.to(device)
    print(x)

def main():
    #creating_tensors()
    #random_tensors()
    #range_of_tensors()
    modifying_tensors()

if __name__ == "__main__":
    main()
