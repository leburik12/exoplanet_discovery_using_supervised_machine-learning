from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


class XGBoostTrainer:
    def __init__(self, scale_pos_weight=None, calibrated=False, calibrator='sigmoid', cv=3):
        self.calibrated = calibrated
        self.calibrator = calibrator
        self.cv = cv

        self.xgb_model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )

        # Wrap XGB with calibration if requested
        if self.calibrated:
            self.model = CalibratedClassifierCV(base_estimator=self.xgb_model, method=self.calibrator, cv=self.cv)
        else:
            self.model = self.xgb_model

    def fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        return self.model.predict(X)

    def feature_importance(self):
        # Feature importance comes from the base XGB estimator
        if hasattr(self.model, "base_estimator"):
            return self.model.base_estimator.feature_importances_
        else:
            return self.model.feature_importances_