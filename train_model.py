import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('mushrooms.csv')

code_to_name = {
    'cap-shape': {'b': 'Bell', 'c': 'Conical', 'x': 'Convex', 'f': 'Flat', 'k': 'Knobbed', 's': 'Sunken'},
    'cap-surface': {'f': 'Fibrous', 'g': 'Grooves', 'y': 'Scaly', 's': 'Smooth'},
    'cap-color': {'n': 'Brown', 'b': 'Buff', 'c': 'Cinnamon', 'g': 'Gray', 'r': 'Green',
                  'p': 'Pink', 'u': 'Purple', 'e': 'Red', 'w': 'White', 'y': 'Yellow'},
    'bruises': {'t': 'True', 'f': 'False'},
    'odor': {'a': 'Almond', 'l': 'Anise', 'c': 'Creosote', 'y': 'Fishy', 'f': 'Foul',
             'm': 'Musty', 'n': 'None', 'p': 'Pungent', 's': 'Spicy'},
    'gill-attachment': {'a': 'Attached', 'd': 'Descending', 'f': 'Free', 'n': 'Notched'},
    'gill-spacing': {'c': 'Close', 'w': 'Crowded', 'd': 'Distant'},
    'gill-size': {'b': 'Broad', 'n': 'Narrow'},
    'gill-color': {'k': 'Black', 'n': 'Brown', 'b': 'Buff', 'h': 'Chocolate', 'g': 'Gray',
                   'r': 'Green', 'o': 'Orange', 'p': 'Pink', 'u': 'Purple', 'e': 'Red',
                   'w': 'White', 'y': 'Yellow'},
    'stalk-shape': {'e': 'Enlarging', 't': 'Tapering'},
    'stalk-root': {'b': 'Bulbous', 'c': 'Club', 'u': 'Cup', 'e': 'Equal',
                   'z': 'Rhizomorphs', 'r': 'Rooted', '?': 'Missing'},
    'stalk-surface-above-ring': {'f': 'Fibrous', 'y': 'Scaly', 'k': 'Silky', 's': 'Smooth'},
    'stalk-surface-below-ring': {'f': 'Fibrous', 'y': 'Scaly', 'k': 'Silky', 's': 'Smooth'},
    'stalk-color-above-ring': {'n': 'Brown', 'b': 'Buff', 'c': 'Cinnamon', 'g': 'Gray',
                               'o': 'Orange', 'p': 'Pink', 'e': 'Red', 'w': 'White', 'y': 'Yellow'},
    'stalk-color-below-ring': {'n': 'Brown', 'b': 'Buff', 'c': 'Cinnamon', 'g': 'Gray',
                               'o': 'Orange', 'p': 'Pink', 'e': 'Red', 'w': 'White', 'y': 'Yellow'},
    'veil-type': {'p': 'Partial', 'u': 'Universal'},
    'veil-color': {'n': 'Brown', 'o': 'Orange', 'w': 'White', 'y': 'Yellow'},
    'ring-number': {'n': 'None', 'o': 'One', 't': 'Two'},
    'ring-type': {'e': 'Evanescent', 'f': 'Flaring', 'l': 'Large', 'n': 'None',
                  'p': 'Pendant', 's': 'Sheathing', 'z': 'Zone'},
    'spore-print-color': {'k': 'Black', 'n': 'Brown', 'b': 'Buff', 'h': 'Chocolate',
                          'r': 'Green', 'o': 'Orange', 'u': 'Purple', 'w': 'White', 'y': 'Yellow'},
    'population': {'a': 'Abundant', 'c': 'Clustered', 'n': 'Numerous',
                   's': 'Scattered', 'v': 'Several', 'y': 'Solitary'},
    'habitat': {'g': 'Grasses', 'l': 'Leaves', 'm': 'Meadows', 'p': 'Paths',
                'u': 'Urban', 'w': 'Waste', 'd': 'Woods'},
    'class': {'e': 'edible', 'p': 'poisonous'}
}

for col, mapping in code_to_name.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

feature_mappings = {}
X = df.drop('class', axis=1)
y = df['class']

for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    feature_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

le_target = LabelEncoder()
y = le_target.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

with open('mushroom_model.pkl', 'wb') as f:
    pickle.dump((model, feature_mappings, le_target, code_to_name), f)

print("✅ Model trained and saved with feature + label mappings.")
