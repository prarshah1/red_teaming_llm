import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('ratings.csv')

# Assuming 'Success' and 'Effort' are numerical columns
plt.figure(figsize=(10, 6))

success_mapping = {
    1: 'No Success',
    2: 'Some Success',
    3: 'Successful',
    4: 'Very Successful'
}

df['success_description'] = df['Success'].map(success_mapping)

# Create the bar plot
plt.bar(df['config_name'], df['Success'])

# Add labels and title
plt.yticks([0, 1, 2, 3, 4], df['success_description'])
plt.xlabel('Configuration Name')
plt.ylabel('Success Level')
plt.title('Success Level by Configuration')

# Display the plot
plt.show()

