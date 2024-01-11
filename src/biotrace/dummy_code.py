import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Create a figure and a grid layout with 3 rows and 2 columns
fig = plt.figure(figsize=(8, 12))
gs = GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])

# Create the first subplot in the first column
ax1 = plt.subplot(gs[0, 0])
ax1.plot([1, 2, 3], [4, 5, 6])
ax1.set_title('Plot 1')

# Create the second subplot in the second column
ax2 = plt.subplot(gs[0, 1])
# Remove the axes for the second subplot
ax2.axis('off')
# Create a table in the second subplot's position
data = [[1, 'Info 1'], [2, 'Info 2'], [3, 'Info 3']]
columns = ('Column 1', 'Column 2')
table = ax2.table(cellText=data, colLabels=columns, cellLoc='center', loc='center')

# Create the third subplot in the first column
ax3 = plt.subplot(gs[1, 0])
ax3.plot([1, 2, 3], [7, 8, 9])
ax3.set_title('Plot 2')

# Create the fourth subplot in the second column
ax4 = plt.subplot(gs[1, 1])
ax4.axis('off')  # Remove the axes for the fourth subplot
table_data = [['A', 'B'], ['C', 'D'], ['E', 'F']]
table_columns = ('Column A', 'Column B')
table2 = ax4.table(cellText=table_data, colLabels=table_columns, cellLoc='center', loc='center')

# Create the fifth subplot in the first column
ax5 = plt.subplot(gs[2, 0])
ax5.plot([1, 2, 3], [10, 11, 12])
ax5.set_title('Plot 3')

# Create the sixth subplot in the second column
ax6 = plt.subplot(gs[2, 1])
ax6.axis('off')  # Remove the axes for the sixth subplot
table_data2 = [['X', 'Y'], ['Z', 'W'], ['P', 'Q']]
table_columns2 = ('Column X', 'Column Y')
table3 = ax6.table(cellText=table_data2, colLabels=table_columns2, cellLoc='center', loc='center')

plt.tight_layout()
plt.show()