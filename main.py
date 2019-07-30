import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import datetime

features = pd.read_csv('./features.csv', index_col='match_id')

for feature in features.columns.values:
    cnt = features[feature].count()
    if cnt != features.shape[0]:
        print feature, cnt

y = features['radiant_win']
X = features.fillna(0).drop(columns=['radiant_win', 'tower_status_radiant', 'tower_status_dire', 'duration',
                                     'barracks_status_radiant', 'barracks_status_dire'])
kf = KFold(n_splits=5, shuffle=True)
for trees in [10, 20, 30]:
    clf = GradientBoostingClassifier(n_estimators=trees)
    start_time = datetime.datetime.now()
    results = cross_val_score(clf, X, y, cv=kf, scoring='roc_auc')
    print 'GradientBoosting time:', datetime.datetime.now() - start_time, 'result', np.mean(results)

clf1 = LogisticRegression(penalty='l2')
start_time = datetime.datetime.now()
results = cross_val_score(clf1, X, y, cv=kf, scoring='roc_auc')
print 'LogisticRegression time:', datetime.datetime.now() - start_time, 'result', np.mean(results)

scaled_X = StandardScaler().fit_transform(X)
for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    start_time = datetime.datetime.now()
    clf1 = LogisticRegression(penalty='l2', C=c)
    results = cross_val_score(clf1, scaled_X, y, cv=kf, scoring='roc_auc')
    print 'Scaled LogisticRegression time:', datetime.datetime.now() - start_time, 'result', np.mean(results)

X_dropped = X.drop(columns=['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'])
scaled_X_dropped = StandardScaler().fit_transform(X_dropped)
for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    start_time = datetime.datetime.now()
    clf1 = LogisticRegression(penalty='l2', C=c)
    results = cross_val_score(clf1, scaled_X_dropped, y, cv=kf, scoring='roc_auc')
    print 'Scaled dropped LogisticRegression time:', datetime.datetime.now() - start_time, 'result', np.mean(results)

N = max(np.unique(X[['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']].values.ravel()))
X_pick = np.zeros((X.shape[0], N))

for i, match_id in enumerate(X.index):
    for p in xrange(5):
        X_pick[i, X.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
        X_pick[i, X.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1

new_X = np.hstack((X_dropped, X_pick))
new_scaled_X = StandardScaler().fit_transform(new_X)
for c in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    start_time = datetime.datetime.now()
    clf1 = LogisticRegression(penalty='l2', C=c)
    results = cross_val_score(clf1, new_scaled_X, y, cv=kf, scoring='roc_auc')
    print 'new scaled LogisticRegression time:', datetime.datetime.now() - start_time, 'result', np.mean(results)

clf2 = LogisticRegression(penalty='l2', C=10)
X_test = pd.read_csv('./features_test.csv', index_col='match_id').fillna(0)
X_test_dropped = X_test.drop(columns=['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero', 'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero'])

X_test_pick = np.zeros((X_test.shape[0], N))

for i, match_id in enumerate(X_test.index):
    for p in xrange(5):
        X_test_pick[i, X_test.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
        X_test_pick[i, X_test.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1

new_test_X = np.hstack((X_test_dropped, X_test_pick))
clf2.fit(new_scaled_X, y)
Y_pred = clf2.predict_proba(StandardScaler().fit_transform(new_test_X))

print min(Y_pred[:, 1]), max(Y_pred[:, 1])
