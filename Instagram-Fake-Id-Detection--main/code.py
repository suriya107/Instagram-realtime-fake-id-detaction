

from google.colab import drive
drive.mount('/content/drive')

import pandas  as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report


data = pd.read_excel('dataset.xlsx')


print(data.info())

data['Column1.isFake'].value_counts()

from sklearn.model_selection import train_test_split
X = data.drop(columns=['Column1.isFake'])
y = data['Column1.isFake']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split





params = {
    'objective': 'binary:logistic',
    'eval_metric': 'error'
}

num_rounds=100;


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
model = xgb.train(params, dtrain, num_rounds)

y_pred = model.predict(dtest)
y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", accuracy)

print("Before OverSampling, counts of label '1': {}".format(sum(y_train == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train == 0)))


from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy={0: 1000, 1: 1000}, random_state=42)
X_t , y_t = sm.fit_resample(X_train, y_train.ravel())



print("After OverSampling, counts of label '1': {}".format(sum(y_t == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_t == 0)))



import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split





params = {
    'objective': 'binary:logistic',
    'eval_metric': 'error'
}

num_rounds=100;


dtrain = xgb.DMatrix(X_t, label=y_t)
dtest = xgb.DMatrix(X_test, label=y_test)
model = xgb.train(params, dtrain, num_rounds)

y_pred = model.predict(dtest)
y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", accuracy)

import pandas as pd
import matplotlib.pyplot as plt

# Assuming y_train is a pandas Series containing your training labels
class_distribution = y_train.value_counts()

# Plot the class distribution
plt.figure(figsize=(8, 6))
class_distribution.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()

# Save the plot as an image file (e.g., PNG)
plt.savefig('class_distribution.png')

# Show the plot (optional)
plt.show()

print("Before OverSampling, counts of label '1': {}".format(sum(y == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y == 0)))


from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy={0: 1000, 1: 1000}, random_state=42)
X_res, y_res = sm.fit_resample(X, y.ravel())



print("After OverSampling, counts of label '1': {}".format(sum(y_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_res == 0)))

import pandas as pd


resampled_data = pd.concat([pd.DataFrame(X_res), pd.DataFrame({'Column1.isFake': y_res})], axis=1)


resampled_data.to_excel('finaldataset.xlsx', index=False)

import pandas as pd
resampled_data = pd.read_excel('finaldataset.xlsx')
X_res = resampled_data.drop('Column1.isFake', axis=1)
y_res = resampled_data['Column1.isFake']
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

import pandas as pd
import matplotlib.pyplot as plt

# Assuming y_train is a pandas Series containing your training labels
class_distribution = y_train.value_counts()

# Plot the class distribution
plt.figure(figsize=(8, 6))
class_distribution.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()

# Save the plot as an image file (e.g., PNG)
plt.savefig('class_distribution.png')

# Show the plot (optional)
plt.show()

import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split





params = {
    'objective': 'binary:logistic',
    'eval_metric': 'error'
}

num_rounds=100;


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
tmodel = xgb.train(params, dtrain, num_rounds)

y_pred = tmodel.predict(dtest)
y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", accuracy)

from sklearn.metrics import f1_score


f1score = f1_score(y_test, y_pred_binary)

print("F1 Score:", f1score)

import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'error'
}

num_rounds = 100


dtrain = xgb.DMatrix(X_train, label=y_train)


cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_rounds,
    nfold=5,
    metrics='error',
    seed=42
)

error_mean = cv_results['test-error-mean'].values[-1]
error_std = cv_results['test-error-std'].values[-1]

print("Mean Error:", error_mean)
print("Standard Deviation of Error:", error_std)


model = xgb.train(params, dtrain, num_boost_round=num_rounds)


dtest = xgb.DMatrix(X_test, label=y_test)
y_pred = model.predict(dtest)
y_pred_binary = [1 if x > 0.5 else 0 for x in y_pred]
accuracy = accuracy_score(y_test, y_pred_binary)
print("Accuracy:", accuracy)

import pandas as pd
import matplotlib.pyplot as plt

# Assuming y_train is a pandas Series containing your training labels
class_distribution = y_train.value_counts()

# Plot the class distribution
plt.figure(figsize=(8, 6))
class_distribution.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()

# Save the plot as an image file (e.g., PNG)
plt.savefig('class_distribution.png')

# Show the plot (optional)
plt.show()

import pandas as pd
from tabulate import tabulate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)

# Create a dictionary to hold the metrics
metrics_dict = {
    'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall'],
    'Value': [accuracy, f1, precision, recall]
}

# Convert the dictionary to a pandas DataFrame
metrics_df = pd.DataFrame(metrics_dict)

# Print the metrics in a table format
print(tabulate(metrics_df, headers='keys', tablefmt='pretty'))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_binary)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_binary)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

pip install instaloader

import instaloader

# Initialize the Instaloader instance
bot = instaloader.Instaloader()

# Login (replace 'your_username' and 'your_password' with your Instagram credentials)
bot.login('mlfakeinsta12345', '******')

# Replace 'username' with the Instagram username you want to retrieve data for
username = 'kerellafollowers'

# Get the user profile
profile = instaloader.Profile.from_username(bot.context, username)

# Print profile information
print("Username: ", profile.username)
print("User Follower Count: ", profile.followers)
print("User Following Count: ", profile.followees)
print("User Biography Length: ", len(profile.biography))
print("User Media Count: ", profile.mediacount)
print("User Has Profile Picture: ", 1 if profile.profile_pic_url else 0)
print("User is Private: ", 1 if profile.is_private else 0)
print("Username Digit Count: ", sum(c.isdigit() for c in profile.username))
print("Username Length: ", len(profile.username))

import instaloader
import csv

# Initialize the Instaloader instance
bot = instaloader.Instaloader()

# Log in to Instagram account
username = 'mlfakeinsta12345'
password = '******'
bot.login(username, password)

# Prompt the user to enter the Instagram username
print("=" * 50)
print("Welcome to Instagram fake account Detector")
print("=" * 50)
target_username = input("\nEnter the Instagram username : ")


# Get the user profile
profile = instaloader.Profile.from_username(bot.context, target_username)

# Create or open a CSV file to store the data
csv_filename = 'instagram.csv'
column_names = [
    'Column1.username',
    'Column1.userFollowerCount',
    'Column1.userFollowingCount',
    'Column1.userBiographyLength',
    'Column1.userMediaCount',
    'Column1.userHasProfilPic',
    'Column1.userIsPrivate',
    'Column1.usernameDigitCount',
    'Column1.usernameLength',
    'Column1.isFake'
]

with open(csv_filename, mode='a', newline='') as file:
    writer = csv.writer(file)

    # Check if the file is empty, if so, write the header row
    if file.tell() == 0:
        writer.writerow(column_names)

    # Write the profile information to the CSV file
    writer.writerow([
        profile.username,
        profile.followers,
        profile.followees,
        len(profile.biography),
        profile.mediacount,
        1 if profile.profile_pic_url else 0,
        1 if profile.is_private else 0,
        sum(c.isdigit() for c in profile.username),
        len(profile.username),
        0
    ])

# Logout from Instagram account

import csv

# Open the CSV file to read the data
csv_filename = 'instagram.csv'

with open(csv_filename, mode='r', newline='') as file:
    reader = csv.reader(file)

    # Read and print each row of the CSV file
    for row in reader:
        print(row)

import pandas as pd
import xgboost as xgb

# Load new data
new_df = pd.read_csv('instagram.csv')

# Assuming 'Column1.username' is the username column
X_new = new_df.drop(['Column1.username', 'Column1.isFake'], axis=1)  # Dropping username and target column

# Convert features into DMatrix
dmatrix_new = xgb.DMatrix(data=X_new)

# Predictions
predictions_new = model.predict(dmatrix_new)

# Printing predictions for each row along with username
for username, prediction in zip(new_df['Column1.username'], predictions_new):
    if prediction > 0.5:
        print(f"{username} is a fake account")
    else:
        print(f"{username} is a real account")

import pandas as pd
import xgboost as xgb

# Load new data
new_df = pd.read_csv('instagram.csv')

# Assuming 'Column1.username' is the username column
X_new = new_df.drop(['Column1.username', 'Column1.isFake'], axis=1)  # Dropping username and target column

# Convert features into DMatrix
dmatrix_new = xgb.DMatrix(data=X_new)

# Predictions
predictions_new = tmodel.predict(dmatrix_new)

# Printing predictions for each row along with username
for username, prediction in zip(new_df['Column1.username'], predictions_new):
    if prediction > 0.5:
        print(f"{username} is a fake account")
    else:
        print(f"{username} is a real account")