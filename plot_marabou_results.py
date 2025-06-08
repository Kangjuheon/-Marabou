import numpy as np
import matplotlib.pyplot as plt

# Adversarial input 시각화
sat_input = np.load('marabou_sat_input.npy')
plt.figure(figsize=(3,3))
plt.imshow(sat_input.squeeze(), cmap='gray')
plt.title('Marabou SAT Input')
plt.axis('off')
plt.savefig('marabou_sat_input.png')
plt.close()

# Output vector 시각화
output_vector = np.load('marabou_output_vector.npy')
plt.figure(figsize=(6,3))
plt.bar(range(len(output_vector)), output_vector)
plt.title('Marabou Output Vector')
plt.xlabel('Class')
plt.ylabel('Output Value')
plt.savefig('marabou_output_vector.png')
plt.close()

print('이미지 저장 완료: marabou_sat_input.png, marabou_output_vector.png') 