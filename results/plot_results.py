import matplotlib.pyplot as plt

models = ["Baseline", "Attention", "LoRA", "Adapter"]

training_time = [5124.92, 2068.53, 2174.67, 2245.16] # seconds

trainable_params = [109000000, 28366848, 294912, 18432]

plt.figure()
plt.bar(models, training_time)
plt.title("Training Time Comparison")
plt.xlabel("Model")
plt.ylabel("Time (seconds)")

plt.savefig("results/training_time.png")
plt.close()

plt.figure()
plt.bar(models, trainable_params)
plt.title("Trainable Parameters Comparison")
plt.xlabel("Model")
plt.ylabel("Number of Parameters")

plt.savefig("results/trainable_params.png")
plt.close()

print("Graphs saved in results/")