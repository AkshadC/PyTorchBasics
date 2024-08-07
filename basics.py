import torch

print(f"Current Torch Version: {torch.__version__}")


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
    random_tensor = torch.randn([3, 3, 3], dtype=torch.float32)
    print(f"Random Tensor : {random_tensor}, \n Shape: {random_tensor.shape}, \n Ndim: {random_tensor.ndim}")

    random_tensor_image = torch.rand(size=(224, 224, 3))
    print(f" Shape: {random_tensor_image.shape}, \n Ndim: {random_tensor_image.ndim}")


def range_of_tensors():
    range_tensor = torch.arange(69, 101, step=4)
    print(f"Range Tensor: {range_tensor}")


def main():
    #creating_tensors()
    #random_tensors()
    range_of_tensors()


if __name__ == "__main__":
    main()
