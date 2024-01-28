from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()
DATA_DIRECTORY = os.getenv("DATA_DIRECTORY")

fig, ax = plt.subplots(figsize=(12, 8))
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
ax.plot(x, y, color="#fc8d62")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('My Plot')
# plt.savefig(f"{DATA_DIRECTORY}/images/p0.png")
# plt.show()

# print(sns.color_palette("Set2").as_hex())