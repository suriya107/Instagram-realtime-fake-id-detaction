WORKING PRINCIPLE

Initial Dataset:

The initial dataset was compiled with information from 1000 genuine accounts and 200 suspicious accounts.

Filename: dataset-withoutoversampling.xlsx

SMOTE Oversampling:

To address the class imbalance issue, the Synthetic Minority Over-sampling Technique (SMOTE) algorithm was employed to augment the dataset.

Final Dataset:

The resulting dataset post oversampling is stored in a file named finaldataset.xlsx.

Dataset Features:

userFollowerCount: Total count of followers for the account (integer)
userFollowingCount: Total count of accounts the user is following (integer)
userBiographyLength: Length of the user's biography text (integer)
userMediaCount: Total number of media posts uploaded by the user (integer)
userHasProfilePic: Binary indicator denoting whether the user has a profile picture (1: yes, 0: no)
userIsPrivate: Binary indicator indicating whether the account is private (1: private, 0: public)
usernameDigitCount: Total count of digits present in the username (integer)
usernameLength: Total length of the username (integer)
isFake: Target variable indicating the account type (1: fake account, 0: real account)
Data Retrieval:

Real-time account data was fetched using Instaloader, a Python library. For comprehensive guidance on Instaloader, refer to its documentation.

Model Usage:

The acquired data was fed into our model to discern whether it corresponds to a genuine or deceptive account.

OUTPUT:

![image](https://github.com/Pranish7D/Instagram-Fake-Id-Detection-/assets/128279506/4384b2c9-87ba-4143-a07c-d77f14f16250)


Now Enter the username you wish to verify:

![image](https://github.com/Pranish7D/Instagram-Fake-Id-Detection-/assets/128279506/ae3579fa-3685-4f90-b504-8042851739dc)
