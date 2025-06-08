import onnx
import numpy as np
from maraboupy import Marabou
from maraboupy import MarabouCore
import torch
import torchvision
import torchvision.transforms as transforms

def load_model():
    # Load the ONNX model
    model = onnx.load("fmnist_cnn.onnx")
    return model

def create_marabou_network(model):
    # Create Marabou network from ONNX model
    network = Marabou.read_onnx(model)
    return network

def get_test_image():
    # Load Fashion MNIST test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                               download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)
    
    # Get a single test image
    image, label = next(iter(test_loader))
    return image.numpy().flatten(), label.item()

def add_input_constraints(network, input_image, epsilon=0.1):
    # Add input constraints for adversarial example
    input_vars = network.inputVars[0]
    
    for i in range(len(input_image)):
        network.setLowerBound(input_vars[i], max(0, input_image[i] - epsilon))
        network.setUpperBound(input_vars[i], min(1, input_image[i] + epsilon))

def add_output_constraints(network, true_label):
    # Add output constraints for targeted misclassification
    output_vars = network.outputVars[0]
    
    # Constraint: true class output should be less than target class output
    for i in range(len(output_vars)):
        if i != true_label:
            network.addInequality([output_vars[true_label], output_vars[i]], [1, -1], 0)

def main():
    # Load model and create Marabou network
    model = load_model()
    network = create_marabou_network(model)
    
    # Get test image
    input_image, true_label = get_test_image()
    
    # Add constraints
    add_input_constraints(network, input_image)
    add_output_constraints(network, true_label)
    
    # Solve the verification problem
    options = Marabou.createOptions(verbosity=0)
    vals, stats = network.solve(options=options)
    
    if vals:
        print("Adversarial example found!")
        adversarial_input = np.array([vals[v] for v in network.inputVars[0]])
        adversarial_input = adversarial_input.reshape(1, 1, 28, 28)
        
        # Save adversarial example
        torch.save(torch.from_numpy(adversarial_input), "adversarial_example.pt")
        np.save("marabou_sat_input.npy", adversarial_input)
        print("Adversarial example saved as 'adversarial_example.pt' and 'marabou_sat_input.npy'")

        # Save output vector (dummy example, replace with actual output if available)
        output_vector = np.array([vals[v] for v in network.outputVars[0]])
        np.save("marabou_output_vector.npy", output_vector)
        print("Output vector saved as 'marabou_output_vector.npy'")
    else:
        print("No adversarial example found within the given constraints.")

if __name__ == "__main__":
    main() 