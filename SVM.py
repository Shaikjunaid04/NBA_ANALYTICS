import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap


# Load the dataset
data = pd.read_csv('Cleaned_SVM.csv')

# Encoding categorical variables
label_encoder = LabelEncoder()
data['team_encoded'] = label_encoder.fit_transform(data['team'])
data['team_opp_encoded'] = label_encoder.transform(data['team_opp'])

# Selecting the specified features
selected_features = data[['team_encoded', 'team_opp_encoded', 'season', 'fg%', '3p%', 'ft%', 
                          'orb', 'drb', 'orb%', 'drb%', 'ft%_max', 'fg%_max', '3p%_max', 
                          'fg%_opp', '3p%_opp', 'ft%_opp', 'orb_opp', 'trb_opp', 'orb%_opp', 
                          'drb%_opp', 'ft%_max_opp', 'fg%_max_opp', '3p%_max_opp']]

# Scaling features
scaler = StandardScaler()
selected_features_scaled = scaler.fit_transform(selected_features)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(selected_features_scaled, data['won'], test_size=0.2, random_state=42)


# Different C values for comparison
c_values = [0.1, 1, 10]
kernels = ['linear', 'rbf', 'poly']

# Dictionaries to store results
accuracies = {}
classification_reports = {}
confusion_matrices = {}

for C in c_values:
    for kernel in kernels:
        # Training the SVM classifier with the specified kernel and C value
        svm_model = SVC(kernel=kernel, C=C)
        svm_model.fit(X_train, y_train)

        # Evaluating the model
        y_pred = svm_model.predict(X_test)
        accuracies[(kernel, C)] = accuracy_score(y_test, y_pred)
        classification_reports[(kernel, C)] = classification_report(y_test, y_pred)
        confusion_matrices[(kernel, C)] = confusion_matrix(y_test, y_pred)

        # Print results
        print(f"C: {C}, Kernel: {kernel}")
        print(f"Accuracy: {accuracies[(kernel, C)]}")
        print("Classification Report:")
        print(classification_reports[(kernel, C)])
        print("Confusion Matrix:")
        print(confusion_matrices[(kernel, C)])
        print("-" * 50)

# Creating a comparison table for accuracies and classification metrics
comparison_factors = {
    "C Value": [],
    "Kernel": [],
    "Accuracy": [],
    "Precision (Class 0)": [],
    "Precision (Class 1)": [],
    "Recall (Class 0)": [],
    "Recall (Class 1)": [],
    "F1-Score (Class 0)": [],
    "F1-Score (Class 1)": []
}

for (kernel, C), report in classification_reports.items():
    report_dict = classification_report(y_test, svm_model.predict(X_test), output_dict=True)
    comparison_factors["C Value"].append(C)
    comparison_factors["Kernel"].append(kernel)
    comparison_factors["Accuracy"].append(accuracies[(kernel, C)])
    comparison_factors["Precision (Class 0)"].append(report_dict['0']['precision'])
    comparison_factors["Precision (Class 1)"].append(report_dict['1']['precision'])
    comparison_factors["Recall (Class 0)"].append(report_dict['0']['recall'])
    comparison_factors["Recall (Class 1)"].append(report_dict['1']['recall'])
    comparison_factors["F1-Score (Class 0)"].append(report_dict['0']['f1-score'])
    comparison_factors["F1-Score (Class 1)"].append(report_dict['1']['f1-score'])

# Converting to DataFrame for better visualization
comparison_df = pd.DataFrame(comparison_factors)
print(comparison_df)


kernels = ['linear', 'rbf', 'poly']
c_values = [0.1, 1, 10]

# Plotting the confusion matrices for each kernel and C value
fig, axes = plt.subplots(len(kernels), len(c_values), figsize=(18, 18))
fig.suptitle('Confusion Matrices for Different Kernels and C values')

for i, kernel in enumerate(kernels):
    for j, C in enumerate(c_values):
        # Training the SVM classifier with the specified kernel and C value
        svm_model = SVC(kernel=kernel, C=C)
        svm_model.fit(X_train, y_train)

        # Predicting on the test set
        y_pred = svm_model.predict(X_test)

        # Computing the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='g', ax=axes[i][j], cmap='Blues')
        axes[i][j].set_title(f'Kernel: {kernel}, C: {C}')
        axes[i][j].set_xlabel('Predicted labels')
        axes[i][j].set_ylabel('True labels')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()




# Sample a subset of the data to reduce computational demand
sample_size = 1000  # Adjust the sample size as needed
data_sample = data.sample(n=sample_size, random_state=42)
X_sample = data_sample[selected_features.columns]
y_sample = data_sample['won']

# Scaling the sampled features
X_sample_scaled = scaler.transform(X_sample)

# Applying PCA to reduce the dataset to 2 dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_sample_scaled)

# Training the SVM model on the 2D PCA transformed data
svm_model_pca = SVC(kernel='poly')
svm_model_pca.fit(X_pca, y_sample)

# Creating a meshgrid for plotting decision boundary
h = .02  # Step size in the mesh
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Plotting decision boundary
plt.figure(figsize=(10, 6))
Z = svm_model_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))

# Plotting the training points
colors = ['red', 'blue']
for i, color in enumerate(colors):
    plt.scatter(X_pca[y_sample == i, 0], X_pca[y_sample == i, 1], c=color, label=f'Class {i}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Decision Boundary of SVM (poly Kernel) after PCA')
plt.legend()

plt.show()



def predict_2(team1, team2, model, label_encoder, scaler, data):    
    mapping = {'ATL': 0, 'BOS': 1, 'BRK': 2, 'CHI': 3, 'CHO': 4, 'CLE': 5, 'DAL': 6, 'DEN': 7, 'DET': 8, 'GSW': 9, 'HOU': 10, 'IND': 11, 'LAC': 12, 'LAL': 13, 'MEM': 14, 'MIA': 15, 'MIL': 16, 'MIN': 17, 'NOP': 18, 'NYK': 19, 'OKC': 20, 'ORL': 21, 'PHI': 22, 'PHO': 23, 'POR': 24, 'SAC': 25, 'SAS': 26, 'TOR': 27, 'UTA': 28, 'WAS': 29}

    team1_encoded = mapping[team1]
    team2_encoded = mapping[team2]
    
    recent_season_data = data[data['season'] == 2022]

    select_cols = ['team_encoded', 'team_opp_encoded', 'season', 'fg%', '3p%', 'ft%', 
                              'orb', 'drb', 'orb%', 'drb%', 'ft%_max', 'fg%_max', '3p%_max', 
                              'fg%_opp', '3p%_opp', 'ft%_opp', 'orb_opp', 'trb_opp', 'orb%_opp', 
                              'drb%_opp', 'ft%_max_opp', 'fg%_max_opp', '3p%_max_opp']
    home = ["season",'team_encoded','fg%','3p%','ft%','orb','drb','orb%','drb%','ft%_max','fg%_max','3p%_max']
    opp = ['team_opp_encoded','fg%_opp','3p%_opp','ft%_opp','orb_opp','trb_opp','orb%_opp','drb%_opp','ft%_max_opp','fg%_max_opp','3p%_max_opp']

    x = recent_season_data[recent_season_data['team_encoded'] == team1_encoded]
    y = recent_season_data[recent_season_data['team_opp_encoded'] == team2_encoded]

    x_mean = x.mean().to_dict()
    y_mean = y.mean().to_dict()
    temp = (pd.concat([pd.DataFrame([x_mean])[home], pd.DataFrame([y_mean])[opp]],axis = 1))
    temp = temp[select_cols]
    feature_vector_scaled = scaler.transform(temp)

    # Predict outcome
    prediction = model.predict(feature_vector_scaled)[0]
    return team1 if prediction else team2


team1 = 'DEN'  # Replace with an actual team name from your dataset
team2 = 'GSW'  # Replace with another actual team name from your dataset
winner = predict_2(team1, team2, svm_model, label_encoder, scaler, data)
print(f"Predicted winner for the teams between {team1} and {team2}: {winner}")
