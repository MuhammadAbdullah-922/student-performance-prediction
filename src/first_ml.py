import pandas as pd

# 1. Data load karo
df = pd.read_csv("data/student-por.csv", sep=";")

# 2. Basic info
print("Shape:", df.shape)
print(df.head())
print(df.info())

# 3. Missing values check
print("Missing values:", df.isnull().sum().sum())

# 4. Target column (G3) ka summary
print(df["G3"].describe())

# 5. Numeric columns ka G3 se correlation
corr = df.corr(numeric_only=True)["G3"].sort_values(ascending=False)
print(corr)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. G1 aur G2 hata do, taake model asal factors seekhe
X = df.drop(columns=["G1", "G2", "G3"])
y = df["G3"]

# 2. Text columns ko numbers me badlo
X = pd.get_dummies(X, drop_first=True)

# 3. Train/test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Model train karo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Test karo
pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
print("R2:", r2_score(y_test, pred))

# 6. Kaun se features sab se important hain
importance = pd.Series(model.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False).head(10))

# Experiment: G1, G2 ke saath
X2 = df.drop(columns=["G3"])
X2 = pd.get_dummies(X2, drop_first=True)

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y, test_size=0.2, random_state=42
)

model2 = RandomForestRegressor(n_estimators=100, random_state=42)
model2.fit(X2_train, y2_train)
pred2 = model2.predict(X2_test)

print("--- G1 aur G2 ke SAATH ---")
print("MAE:", mean_absolute_error(y2_test, pred2))
print("R2:", r2_score(y2_test, pred2))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Pass/Fail column banao
df["pass"] = (df["G3"] >= 10).astype(int)

# 2. Input (X) aur jawab (y)
Xc = df.drop(columns=["G2", "G3", "pass"])
Xc = pd.get_dummies(Xc, drop_first=True)
yc = df["pass"]

# 3. Train/test split
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, random_state=42
)

# 4. Model train karo
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
clf.fit(Xc_train, yc_train)

# 5. Test karo
pred_c = clf.predict(Xc_test)
print("--- PASS/FAIL ---")
print("Accuracy:", accuracy_score(yc_test, pred_c))
print(confusion_matrix(yc_test, pred_c))

from sklearn.metrics import classification_report
print(classification_report(yc_test, pred_c))

